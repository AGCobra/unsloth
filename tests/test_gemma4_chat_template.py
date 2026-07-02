import os
import re

import pytest
from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import TemplateError


CHAT_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "chat_templates.py",
)

# 12B / 26B-A4B / 31B convention: empty thought block prefilled when thinking off.
NON_EDGE = "gemma4_template"
# E2B / E4B convention: the empty thought block is never emitted.
EDGE = "gemma4_edge_template"

EMPTY_BLOCK = "<|channel>thought\n<channel|>"


def _extract_template(name):
    src = open(CHAT_TEMPLATES_PATH).read()
    pattern = rf'{re.escape(name)}\s*=\s*\\\n"""(.*?)"""'
    m = re.search(pattern, src, flags = re.DOTALL)
    assert m, f"Could not extract {name} from chat_templates.py"
    return m.group(1)


def _env():
    env = Environment(undefined = StrictUndefined, trim_blocks = False, lstrip_blocks = False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(TemplateError(msg))
    return env


def _render(template_name, messages, **kwargs):
    src = _extract_template(template_name)
    tmpl = _env().from_string(src)
    ctx = {"messages": messages, "add_generation_prompt": False}
    ctx.update(kwargs)
    return tmpl.render(**ctx)


# ---------- system turn and <|think|> placement ----------


def test_system_message_emits_dedicated_system_turn():
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
    ]
    out = _render(NON_EDGE, msgs)
    assert "<|turn>system\nYou are helpful<turn|>" in out
    assert "<|turn>user\nHi<turn|>" in out
    assert "You are helpful\n\nHi" not in out


def test_developer_role_treated_as_system():
    msgs = [
        {"role": "developer", "content": "Internal instructions"},
        {"role": "user", "content": "Hi"},
    ]
    out = _render(NON_EDGE, msgs)
    assert "<|turn>system\nInternal instructions<turn|>" in out


def test_no_system_no_thinking_has_no_system_turn():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(NON_EDGE, msgs)
    assert "<|turn>user\nHi<turn|>" in out
    assert "<|turn>system" not in out


@pytest.mark.parametrize("tpl", [NON_EDGE, EDGE])
def test_thinking_defaults_off_no_think_token(tpl):
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(tpl, msgs)
    assert "<|think|>" not in out


@pytest.mark.parametrize("tpl", [NON_EDGE, EDGE])
def test_think_token_emitted_in_system_turn_when_enabled(tpl):
    msgs = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hi"}]
    out = _render(tpl, msgs, enable_thinking = True)
    assert "<|turn>system\n<|think|>\nSys<turn|>" in out


def test_alternation_violation_raises_template_error():
    msgs = [{"role": "user", "content": "A"}, {"role": "user", "content": "B"}]
    with pytest.raises(TemplateError):
        _render(NON_EDGE, msgs)


# ---------- SFT target must extend the inference prompt exactly ----------
# The rendered training text for (prompt + assistant answer) must start with the
# rendered inference prompt for the same messages, so SFT conditions the answer
# on exactly what the model sees at generation time.


@pytest.mark.parametrize("tpl", [NON_EDGE, EDGE])
def test_sft_target_extends_inference_prompt_thinking_off(tpl):
    msgs = [{"role": "user", "content": "Hi"}]
    prompt = _render(tpl, msgs, add_generation_prompt = True)
    full = _render(tpl, msgs + [{"role": "assistant", "content": "Hello!"}])
    assert full.startswith(prompt)
    assert full[len(prompt):] == "Hello!<turn|>\n"


@pytest.mark.parametrize("tpl", [NON_EDGE, EDGE])
def test_sft_target_extends_inference_prompt_thinking_on(tpl):
    msgs = [{"role": "user", "content": "Hi"}]
    prompt = _render(tpl, msgs, add_generation_prompt = True, enable_thinking = True)
    tgt = {"role": "assistant", "content": "<|channel>thought\nhmm<channel|>Hello!"}
    full = _render(tpl, msgs + [tgt], enable_thinking = True)
    assert full.startswith(prompt)
    assert full[len(prompt):] == "<|channel>thought\nhmm<channel|>Hello!<turn|>\n"


@pytest.mark.parametrize("tpl", [NON_EDGE, EDGE])
def test_sft_target_extends_inference_prompt_multi_turn(tpl):
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    prompt = _render(tpl, msgs, add_generation_prompt = True)
    full = _render(tpl, msgs + [{"role": "assistant", "content": "A2"}])
    assert full.startswith(prompt)
    assert full[len(prompt):] == "A2<turn|>\n"


# ---------- empty thought block placement (12B / 26B-A4B / 31B) ----------


def test_non_edge_final_model_turn_has_empty_block_thinking_off():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    out = _render(NON_EDGE, msgs)
    assert out.endswith("<|turn>model\n<|channel>thought\n<channel|>A<turn|>\n")


def test_non_edge_generation_prompt_prefills_empty_block_thinking_off():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(NON_EDGE, msgs, add_generation_prompt = True)
    assert out.endswith("<|turn>model\n<|channel>thought\n<channel|>")


def test_non_edge_generation_prompt_bare_when_thinking_on():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(NON_EDGE, msgs, add_generation_prompt = True, enable_thinking = True)
    assert out.endswith("<|turn>model\n")
    assert "<|channel>" not in out


def test_historical_model_turns_stripped_and_blockless():
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<|channel>r1<channel|>A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<|channel>r2<channel|>A2"},
    ]
    out = _render(NON_EDGE, msgs)
    assert "r1" not in out and "r2" not in out
    assert "<|turn>model\nA1<turn|>" in out
    # Only the final model turn carries the (empty) thought block.
    assert out.count("<|channel>thought") == 1
    assert out.endswith("<|channel>thought\n<channel|>A2<turn|>\n")


def test_inference_history_rendered_same_as_training_history():
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    out = _render(NON_EDGE, msgs, add_generation_prompt = True)
    assert "<|turn>model\nA1<turn|>" in out
    assert out.count("<|channel>thought") == 1  # only the generation-prompt prefill
    assert out.endswith("<|turn>model\n<|channel>thought\n<channel|>")


# ---------- edge (E2B / E4B) never emits the empty block ----------


def test_edge_final_model_turn_has_no_block_thinking_off():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    out = _render(EDGE, msgs)
    assert out.endswith("<|turn>model\nA<turn|>\n")
    assert EMPTY_BLOCK not in out


def test_edge_generation_prompt_is_always_bare():
    msgs = [{"role": "user", "content": "Hi"}]
    off = _render(EDGE, msgs, add_generation_prompt = True)
    on = _render(EDGE, msgs, add_generation_prompt = True, enable_thinking = True)
    assert off.endswith("<|turn>model\n")
    assert on.endswith("<|turn>model\n")
    assert "<|channel>" not in off and "<|channel>" not in on


# ---------- thinking-on SFT keeps the final reasoning trace ----------


@pytest.mark.parametrize("tpl", [NON_EDGE, EDGE])
def test_final_turn_embedded_trace_kept_when_thinking_on(tpl):
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "<|channel>thought\n2+2=4<channel|>The answer is 4."},
    ]
    out = _render(tpl, msgs, enable_thinking = True)
    assert out.endswith("<|turn>model\n<|channel>thought\n2+2=4<channel|>The answer is 4.<turn|>\n")


@pytest.mark.parametrize("tpl", [NON_EDGE, EDGE])
def test_reasoning_field_rendered_as_thought_channel_when_thinking_on(tpl):
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A", "reasoning_content": "hmm"},
    ]
    out = _render(tpl, msgs, enable_thinking = True)
    assert out.endswith("<|turn>model\n<|channel>thought\nhmm\n<channel|>A<turn|>\n")


def test_reasoning_field_dropped_when_thinking_off():
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A", "reasoning_content": "hmm"},
    ]
    out = _render(NON_EDGE, msgs)
    assert "hmm" not in out
    assert out.endswith("<|channel>thought\n<channel|>A<turn|>\n")


# ---------- strip_thinking semantics ----------


def test_final_turn_trace_stripped_when_thinking_off():
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "<|channel>thought\n2+2=4<channel|>The answer is 4."},
    ]
    out = _render(NON_EDGE, msgs)
    assert "2+2=4" not in out
    assert out.endswith("<|channel>thought\n<channel|>The answer is 4.<turn|>\n")


def test_strip_thinking_preserves_plain_text():
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "plain answer with no markup"},
    ]
    out = _render(NON_EDGE, msgs)
    assert "plain answer with no markup" in out


def test_iterable_text_final_turn_gets_block_thinking_off():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Q"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|channel>r<channel|>final"}],
        },
    ]
    out = _render(NON_EDGE, msgs)
    assert "<|channel>r" not in out
    assert out.endswith("<|channel>thought\n<channel|>final<turn|>\n")
