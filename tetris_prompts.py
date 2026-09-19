"""Prompt and hypothesis builders for openjev Tetris."""
from typing import Dict, List, Tuple
from tetris_env import Tetris, compute_board_features

PROMPT_STRATEGIES = ["qualitative", "qualitative_fine", "outcome_entailment", "coach_rule", "action_ranking", "sign"]

_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


def _number_word(value: int) -> str:
    """Spell small board values so NLI can use learned word semantics, not digits."""
    value = int(value)
    return _NUMBER_WORDS.get(value, "many")


def get_move_descriptors(env: Tetris, rot: int, col: int, curr_feats: Dict | None = None) -> Dict:
    curr_feats = curr_feats if curr_feats is not None else env.state()["features"]
    sim_grid, lines, landing_h = env.simulate_placement(rot, col)
    sim_feats = compute_board_features(sim_grid)
    delta_holes = sim_feats["holes"] - curr_feats["holes"]
    delta_max_h = sim_feats["max_height"] - curr_feats["max_height"]

    return {
        "rot": rot,
        "col": col,
        "piece": env.current_piece,
        "lines": lines,
        "landing_h": landing_h,
        "sim_feats": sim_feats,
        "curr_feats": curr_feats,
        "delta_holes": delta_holes,
        "delta_max_h": delta_max_h,
    }


def build_pair(desc: Dict, strategy: str = "outcome_entailment") -> Tuple[str, str]:
    piece = desc["piece"]
    rot = desc["rot"]
    col = desc["col"]
    lines = desc["lines"]
    landing_h = desc["landing_h"]
    delta_holes = desc["delta_holes"]
    sim_feats = desc["sim_feats"]
    curr_feats = desc["curr_feats"]

    if strategy == "qualitative":
        # An NLI model judges words, not arithmetic, so the outcome is rendered as plain
        # English. But the phrasing also has to SEPARATE moves: with coarse buckets 23
        # legal placements collapsed into 5 distinct premises and 78% of moves tied,
        # so each attribute is graded finely enough to keep candidates distinguishable.
        max_h = sim_feats["max_height"]
        bump = sim_feats["bumpiness"]
        delta_wells = sim_feats["wells"] - curr_feats["wells"]

        if lines >= 4:
            line_txt = "It clears four lines at once."
        elif lines > 1:
            line_txt = f"It clears {['', '', 'two', 'three'][lines]} lines at once."
        elif lines == 1:
            line_txt = "It completes a full line."
        else:
            line_txt = "It completes no line."

        if delta_holes <= 0:
            hole_txt = "It traps no empty square underneath it."
        elif delta_holes == 1:
            hole_txt = "It seals one empty square underneath it, which is wasted space."
        elif delta_holes == 2:
            hole_txt = "It seals a pair of empty squares underneath it, which is wasted space."
        else:
            hole_txt = "It buries many empty squares underneath it, which is badly wasted space."

        land_txt = ("It settles right down on the floor." if landing_h <= 2 else
                    "It settles low down on the stack." if landing_h <= 5 else
                    "It rests partway up the stack." if landing_h <= 9 else
                    "It perches high up on the stack." if landing_h <= 13 else
                    "It perches near the very top of the board.")

        height_txt = ("The board stays almost empty." if max_h <= 3 else
                      "The board stays low and safe." if max_h <= 6 else
                      "The board is filling up." if max_h <= 10 else
                      "The board is getting worryingly full." if max_h <= 14 else
                      "The board is dangerously close to the top.")

        flat_txt = ("The surface is left perfectly even." if bump <= 3 else
                    "The surface is left fairly even." if bump <= 7 else
                    "The surface is left bumpy." if bump <= 12 else
                    "The surface is left very jagged and uneven.")

        well_txt = ("" if delta_wells <= 0 else
                    " It opens a narrow gap beside it." if delta_wells <= 3 else
                    " It opens a deep narrow canyon beside it.")

        premise = (f"Tetris outcome: a {piece} piece is dropped into the board. {line_txt} {hole_txt} "
                   f"{land_txt}{well_txt} {height_txt} {flat_txt}")
        hypothesis = "This is a good, safe move."
    elif strategy == "qualitative_fine":
        # A compact wording for a learned policy. Every scored feature is represented
        # by a distinct, human-readable value; digits are deliberately avoided because
        # an NLI model does not reliably compare arbitrary numbers.
        delta_wells = sim_feats["wells"] - curr_feats["wells"]
        premise = (
            f"Tetris move outcome. The {piece} piece at column {_number_word(col)} "
            f"with rotation {_number_word(rot)} clears {_number_word(lines)} lines, "
            f"adds {_number_word(max(delta_holes, 0))} covered holes, and lands at "
            f"height {_number_word(landing_h)}. The resulting stack is "
            f"{_number_word(sim_feats['max_height'])} blocks high with "
            f"{_number_word(sim_feats['bumpiness'])} surface roughness and "
            f"{_number_word(max(delta_wells, 0))} new wells."
        )
        hypothesis = "This is a strong, safe Tetris move."
    elif strategy == "outcome_entailment":
        premise = (
            f"Tetris game state. Current piece to drop is {piece}. "
            f"Placing the piece at column {col} with rotation {rot} results in landing height {landing_h}. "
            f"Lines cleared: {lines}. New holes created: {delta_holes}. Total covered holes: {sim_feats['holes']}. "
            f"Resulting maximum stack height: {sim_feats['max_height']}. Surface bumpiness: {sim_feats['bumpiness']}."
        )
        hypothesis = (
            "The piece is placed cleanly and safely, clearing lines without creating trapped holes under the stack."
        )
    elif strategy == "coach_rule":
        premise = (
            f"Tetris strategy rule: keep the board low and flat, clear complete lines, and avoid trapping empty holes. "
            f"Current board has max height {curr_feats['max_height']} and {curr_feats['holes']} holes. "
            f"Dropping piece {piece} at column {col} with rotation {rot} clears {lines} line(s), creates {delta_holes} new holes, "
            f"and lands at height {landing_h} (resulting max height {sim_feats['max_height']})."
        )
        hypothesis = (
            "This placement follows the strategy rule by avoiding trapped holes and keeping the stack low."
        )
    elif strategy == "sign":
        # Direct polar statements that reward 0 holes and line clears
        premise = (
            f"Tetris move simulation. Piece: {piece}, column: {col}, rotation: {rot}. "
            f"Lines cleared: {lines}. Holes added: {delta_holes}. Final height: {landing_h}."
        )
        if delta_holes == 0:
            hypothesis = "The placement creates zero covered holes."
        else:
            hypothesis = f"The placement creates {delta_holes} covered holes."
    elif strategy == "action_ranking":
        premise = (
            f"Tetris board decision. Current piece: {piece}. Stack height: {curr_feats['max_height']}, holes: {curr_feats['holes']}. "
            f"Target placement: column {col}, rotation {rot} (clears {lines} lines, creates {delta_holes} new holes)."
        )
        hypothesis = f"The optimal placement is to place the {piece} at column {col} with rotation {rot}."
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return premise, hypothesis


def generate_candidate_pairs(env: Tetris, legal_moves: List[Tuple[int, int]], strategy: str = "outcome_entailment") -> List[Tuple[str, str]]:
    # state() computes the current feature vector. It is shared by every candidate.
    # Reusing it matters when this function is called for all rotations and columns.
    curr_feats = env.state()["features"]
    pairs = []
    for rot, col in legal_moves:
        desc = get_move_descriptors(env, rot, col, curr_feats=curr_feats)
        pairs.append(build_pair(desc, strategy=strategy))
    return pairs
