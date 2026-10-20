import difflib
import hashlib
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C8_SOURCE_PATH = ROOT / "physbench/r5/m1_c8_reproduction.py"
R5_SOURCE_PATH = ROOT / "physbench/r5/m1_r5_formal.py"
D1_SCRIPT_PATH = ROOT / "physbench/r5/m1_r5_diag_d1_v2.py"

def extract_between(text: str, start_str: str, end_str: str) -> str:
    p1 = text.find(start_str)
    if p1 == -1:
        raise ValueError(f"Start delimiter not found: {start_str!r}")
    p2 = text.find(end_str, p1)
    if p2 == -1:
        raise ValueError(f"End delimiter not found: {end_str!r}")
    return text[p1:p2 + len(end_str)].strip()

def extract_copy_block(text: str, block_name: str) -> str:
    start_tag = f"# >>> COPY_BLOCK: {block_name}"
    end_tag = f"# <<< COPY_BLOCK: {block_name}"
    p1 = text.find(start_tag)
    if p1 == -1:
        raise ValueError(f"Start tag not found: {start_tag}")
    p1 += len(start_tag)
    p2 = text.find(end_tag, p1)
    if p2 == -1:
        raise ValueError(f"End tag not found: {end_tag}")
    raw = text[p1:p2]
    return textwrap.dedent(raw).strip()

def run_copy_gate():
    print("=" * 70)
    print("RUNNING STATIC COPY GATE FOR D1-V2")
    print("=" * 70)

    if not C8_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing C8 source: {C8_SOURCE_PATH}")
    if not R5_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing R5 source: {R5_SOURCE_PATH}")
    if not D1_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing D1-v2 script: {D1_SCRIPT_PATH}")

    c8_text = C8_SOURCE_PATH.read_text(encoding="utf-8")
    r5_text = R5_SOURCE_PATH.read_text(encoding="utf-8")
    d1_text = D1_SCRIPT_PATH.read_text(encoding="utf-8")

    # 1. c8_env
    b_env_src = extract_between(
        c8_text,
        "env=gym.make('gl_gym/GreenLightTomato-v0',",
        "assert abs(P_HV[55]/P_HV[46]-VENT_PROJECTED_RATIO)<=1e-12"
    )

    # 2. c8_build_weather
    b_weather_src = extract_between(
        c8_text,
        "def build_weather(r):",
        "    return d"
    )

    # 3. c8_full_action
    b_action_src = extract_between(
        c8_text,
        "def full_action(v):",
        "return np.array([0,0,0,float(v),0,0],dtype=float)"
    )

    # 4. c8_initial_state
    b_init_src = extract_between(
        c8_text,
        "def initial_state(r):",
        "    return xx"
    )

    # 5. c8_symbolic (allowed rename: FCF -> FCF_C8)
    b_c8_sym_raw = extract_between(
        c8_text,
        "x=ca.SX.sym('x',NX)",
        "{'abstol':1e-5,'reltol':1e-5,'max_num_steps':90000}\n)"
    )
    b_c8_sym_src = b_c8_sym_raw.replace("FCF=ca.integrator(", "FCF_C8=ca.integrator(")

    # 6. r5_symbolic (allowed rename: FCF -> FCF_R5, de-indented by 4 spaces)
    b_r5_sym_raw = extract_between(
        r5_text,
        'x = ca.SX.sym("x", NX)',
        '{"abstol": 1e-5, "reltol": 1e-5, "max_num_steps": 90000})'
    )
    b_r5_sym_src = "\n".join(
        [line[4:] if line.startswith("    ") else line for line in b_r5_sym_raw.splitlines()]
    ).replace("FCF = ca.integrator(", "FCF_R5 = ca.integrator(")

    # 7. r5_pvec_call (allowed rename: FCF -> FCF_R5, de-indented by 4 spaces)
    b_r5_call_raw = extract_between(
        r5_text,
        "pvec = np.concatenate([dd, P_HV, [qt, qm, hfac]])",
        'dv900 = float(np.asarray(rr["qf"].full()).reshape(-1)[0])'
    )
    b_r5_call_src = "\n".join(
        [line[4:] if line.startswith("    ") else line for line in b_r5_call_raw.splitlines()]
    ).replace("rr = FCF(", "rr = FCF_R5(")

    expected_blocks = {
        "c8_env": b_env_src,
        "c8_build_weather": b_weather_src,
        "c8_full_action": b_action_src,
        "c8_initial_state": b_init_src,
        "c8_symbolic": b_c8_sym_src,
        "r5_symbolic": b_r5_sym_src,
        "r5_pvec_call": b_r5_call_src,
    }

    all_passed = True
    block_hashes = {}

    for name, expected_content in expected_blocks.items():
        try:
            actual_content = extract_copy_block(d1_text, name)
        except Exception as ex:
            print(f"[FAIL] Block '{name}': {ex}")
            all_passed = False
            continue

        exp_lines = expected_content.splitlines(keepends=True)
        act_lines = actual_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            exp_lines, act_lines,
            fromfile=f"frozen_source::{name}",
            tofile=f"d1_v2_script::{name}"
        ))

        h = hashlib.sha256(actual_content.encode("utf-8")).hexdigest()
        block_hashes[name] = h

        if diff:
            print(f"[FAIL] Block '{name}' differs from frozen source:")
            sys.stdout.writelines(diff)
            all_passed = False
        else:
            print(f"[PASS] Block '{name:16s}' SHA256: {h}")

    if not all_passed:
        print("\nSTATIC COPY GATE FAILED: Code deviations detected.")
        sys.exit(1)

    print("\nSTATIC COPY GATE PASS: All blocks identical to frozen sources with allowed renames.")
    return block_hashes

if __name__ == "__main__":
    run_copy_gate()
