"""serve/test_minimax_tools.py - MiniMax-M2's tools and tool calls.

The M2 template takes the tools in OpenAI's wrapper and writes its calls as
`<minimax:tool_call><invoke name="..."><parameter name="...">...`.  ChatTemplate adapts the tools it renders and
OutputParser translates the calls to its own form, so the parser, its streaming events and the API writers are the
same as for the Qwen template.  These tests pin both halves, including a real M2 template from the tree.

    python -m unittest serve.test_minimax_tools -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import (  # noqa: E402
    ChatTemplate, OutputParser, forced_call, openai_to_messages, parse_tool_call,
)
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M2_TEMPLATE = ROOT / "third_party/llama.cpp/models/templates/MiniMax-M2.jinja"
QWEN_TEMPLATE = ROOT / "serve/chat_template.jinja"
CTX = 32768

TOOLS = [{"name": "read_file", "description": "read a file",
          "parameters": {"type": "object", "properties": {
              "limit": {"type": "integer"}, "path": {"type": "string"}}}}]
CALL = ('<minimax:tool_call>\n<invoke name="read_file">\n'
        '<parameter name="limit">120</parameter>\n<parameter name="path">file.txt</parameter>\n'
        '</invoke>\n</minimax:tool_call>')
INVOKE = ('<invoke name="read_file">\n<parameter name="limit">120</parameter>\n'
          '<parameter name="path">file.txt</parameter>\n</invoke>')


def run(text: str, width: int, tools=TOOLS, thinking: bool = False, stream_tools: bool = False):
    parser = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools, tool_format="minimax")
    events = []
    for i in range(0, len(text), width):
        events += parser.feed(text[i:i + width])
    return events + parser.finish()


def joined(events, kind: str) -> str:
    return "".join(event.text for event in events if event.kind == kind)


class ToolShape(unittest.TestCase):
    def test_the_m2_template_takes_openai_tools(self):
        template = ChatTemplate(M2_TEMPLATE)
        self.assertEqual((template.tool_format, template.tools_wrapped), ("minimax", True))
        self.assertIs(template.caps["supports_tools"], True)
        self.assertIs(template.caps["supports_tool_calls"], True)
        self.assertIs(template.caps["supports_parallel_tool_calls"], True)
        # M2 keeps an assistant's reasoning only after the last user turn (its template's rule), so the probe's
        # older assistant turn is not preserved: the hint is off, as for any template with that rule.
        self.assertIs(template.caps["supports_preserve_reasoning"], False)
        messages, tools, _ = openai_to_messages({
            "messages": [{"role": "user", "content": "read file.txt"}],
            "tools": [{"type": "function", "function": TOOLS[0]}]})
        prompt = template.render(messages, tools=tools)
        self.assertIn('<tool>{"name": "read_file"', prompt)   # the inner function object, once
        self.assertIn("<minimax:tool_call>", prompt)

    def test_the_qwen_template_still_takes_bare_tools(self):
        template = ChatTemplate(QWEN_TEMPLATE)
        self.assertEqual((template.tool_format, template.tools_wrapped), ("qwen", False))
        messages, tools, _ = openai_to_messages({
            "messages": [{"role": "user", "content": "read file.txt"}],
            "tools": [{"type": "function", "function": TOOLS[0]}]})
        prompt = template.render(messages, tools=tools)
        self.assertIn('"name": "read_file"', prompt)
        self.assertNotIn("<minimax:tool_call>", prompt)
        for key in ("supports_tools", "supports_tool_calls", "supports_system_role",
                    "supports_parallel_tool_calls", "supports_preserve_reasoning"):
            self.assertIs(template.caps[key], True, key)


class ParseInvoke(unittest.TestCase):
    def test_values_follow_the_schema(self):
        call = parse_tool_call(INVOKE, TOOLS[0])
        self.assertEqual((call.name, call.arguments), ("read_file", {"limit": 120, "path": "file.txt"}))

    def test_a_non_string_value_is_json(self):
        tool = {"name": "configure", "parameters": {"properties": {"opts": {"type": "object"}}}}
        call = parse_tool_call('<invoke name="configure"><parameter name="opts">{"a": 1}</parameter></invoke>',
                               tool)
        self.assertEqual(call.arguments, {"opts": {"a": 1}})

    def test_one_newline_around_a_value_is_dropped(self):
        call = parse_tool_call('<invoke name="read_file">\n<parameter name="path">\nfile.txt\n</parameter>\n'
                               '</invoke>', TOOLS[0])
        self.assertEqual(call.arguments, {"path": "file.txt"})

    def test_a_malformed_invoke_is_a_value_error(self):
        for body in ('<invoke name="read_file"><parameter name="path">file.txt</invoke>',
                     '<invoke read_file></invoke>'):
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    parse_tool_call(body, TOOLS[0])


class Parser(unittest.TestCase):
    def test_a_call_in_the_answer(self):
        for width in (1, 2, 3, 7, 64, 100_000):
            with self.subTest(width=width):
                events = run("ok " + CALL + " done", width)
                calls = [event.call for event in events if event.kind == "tool_call"]
                self.assertEqual([(call.name, call.arguments) for call in calls],
                                 [("read_file", {"limit": 120, "path": "file.txt"})])
                content = joined(events, "content")
                self.assertIn("ok", content)
                self.assertIn("done", content)
                self.assertNotIn("minimax", content)
                self.assertNotIn('<parameter name="', content)

    def test_the_call_is_streamed_as_its_arguments_arrive(self):
        events = run(CALL, 5, stream_tools=True)
        starts = [event for event in events if event.kind == "tool_start"]
        self.assertEqual([event.call.name for event in starts], ["read_file"])
        streamed = "".join(event.text for event in events if event.kind == "tool_args")
        self.assertEqual(json.loads(streamed), {"limit": 120, "path": "file.txt"})
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual([(call.name, call.arguments) for call in calls],
                         [("read_file", {"limit": 120, "path": "file.txt"})])

    def test_a_call_at_the_start(self):
        for width in (1, 3, 100_000):
            with self.subTest(width=width):
                calls = [event.call for event in run(CALL, width) if event.kind == "tool_call"]
                self.assertEqual([(call.name, call.arguments) for call in calls],
                                 [("read_file", {"limit": 120, "path": "file.txt"})])

    def test_an_invoke_without_parameters(self):
        text = '<minimax:tool_call>\n<invoke name="read_file"></invoke>\n</minimax:tool_call>'
        for width in (1, 5, 100_000):
            with self.subTest(width=width):
                calls = [event.call for event in run(text, width) if event.kind == "tool_call"]
                self.assertEqual([(call.name, call.arguments) for call in calls], [("read_file", {})])

    def test_a_non_string_value_in_the_stream(self):
        tool = {"name": "configure", "parameters": {"properties": {"opts": {"type": "object"}}}}
        text = ('<minimax:tool_call>\n<invoke name="configure">\n'
                '<parameter name="opts">{"a": [1, true]}</parameter>\n'
                '</invoke>\n</minimax:tool_call>')
        for width in (1, 4, 100_000):
            with self.subTest(width=width):
                calls = [event.call for event in run(text, width, tools=[tool]) if event.kind == "tool_call"]
                self.assertEqual([(call.name, call.arguments) for call in calls],
                                 [("configure", {"opts": {"a": [1, True]}})])

    def test_two_invokes_in_one_envelope(self):
        text = ('<minimax:tool_call>\n'
                '<invoke name="read_file"><parameter name="path">a.txt</parameter></invoke>\n'
                '<invoke name="read_file"><parameter name="path">b.txt</parameter></invoke>\n'
                '</minimax:tool_call>')
        for width in (1, 3, 100_000):
            with self.subTest(width=width):
                calls = [event.call for event in run(text, width) if event.kind == "tool_call"]
                self.assertEqual([(call.name, call.arguments) for call in calls],
                                 [("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"})])

    def test_a_value_may_contain_the_call_wording(self):
        text = ('<minimax:tool_call>\n<invoke name="read_file">\n'
                '<parameter name="path">x<invoke name="evil">y</parameter>\n'
                '</invoke>\n</minimax:tool_call>')
        for width in (1, 5, 100_000):
            with self.subTest(width=width):
                events = run(text, width)
                calls = [event.call for event in events if event.kind == "tool_call"]
                self.assertEqual([(call.name, call.arguments) for call in calls],
                                 [("read_file", {"path": 'x<invoke name="evil">y'})])

    def test_a_value_may_contain_a_parameter_tag(self):
        text = ('<minimax:tool_call>\n<invoke name="read_file">\n'
                '<parameter name="path">x<parameter name="p">y</parameter>\n'
                '</invoke>\n</minimax:tool_call>')
        events = run(text, 3)
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual([(call.name, call.arguments) for call in calls],
                         [("read_file", {"path": 'x<parameter name="p">y'})])

    def test_without_tools_the_wording_is_left_alone(self):
        for width in (1, 5, 100_000):
            with self.subTest(width=width):
                events = run(CALL, width, tools=None)
                self.assertFalse([event for event in events if event.kind == "tool_call"])
                self.assertEqual(joined(events, "content"), CALL)

    def test_a_call_inside_the_reasoning(self):
        text = "before\n" + CALL + "\n</think>\n\nanswer"
        for width in (1, 2, 7, 100_000):
            with self.subTest(width=width):
                events = run(text, width, thinking=True)
                calls = [event.call for event in events if event.kind == "tool_call"]
                self.assertEqual([(call.name, call.arguments) for call in calls],
                                 [("read_file", {"limit": 120, "path": "file.txt"})])
                self.assertEqual(joined(events, "reasoning"), "before\n")
                self.assertEqual(joined(events, "content"), "answer")

    def test_an_unfinished_call_is_text(self):
        text = "before\n<minimax:tool_call>\n<invoke name=\"read_file\">\n<parameter name=\"path\">file"
        for width in (1, 7, 100_000):
            with self.subTest(width=width):
                events = run(text, width)
                self.assertFalse([event for event in events if event.kind == "tool_call"])
                self.assertIn("file", joined(events, "content"))

    def test_plain_text_keeps_its_angle_brackets(self):
        text = "2 < 3 and <div>x</div>, nothing else"
        for width in (1, 4, 100_000):
            with self.subTest(width=width):
                events = run(text, width)
                self.assertEqual(joined(events, "content"), text)

    def test_the_qwen_wording_still_parses(self):
        text = ("<tool_call>\n<function=read_file>\n<parameter=limit>\n120\n</parameter>\n"
                "<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>")
        events = run(text, 3)
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual([(call.name, call.arguments) for call in calls],
                         [("read_file", {"limit": 120, "path": "file.txt"})])


class ForcedCall(unittest.TestCase):
    def test_qwen_is_unchanged(self):
        two = TOOLS + [{"name": "write_file", "parameters": {"properties": {"path": {"type": "string"}}}}]
        self.assertEqual(forced_call("required", two), "<tool_call>\n<function=")
        self.assertEqual(forced_call("required", TOOLS), "<tool_call>\n<function=read_file>\n")
        self.assertEqual(forced_call({"type": "function", "function": {"name": "read_file"}}, TOOLS),
                         "<tool_call>\n<function=read_file>\n")

    def test_minimax_opens_its_envelope(self):
        self.assertEqual(forced_call({"type": "function", "function": {"name": "read_file"}}, TOOLS, "minimax"),
                         '<minimax:tool_call>\n<invoke name="read_file">\n')
        two = TOOLS + [{"name": "write_file", "parameters": {"properties": {"path": {"type": "string"}}}}]
        self.assertEqual(forced_call("required", two, "minimax"), '<minimax:tool_call>\n<invoke name="')

    def test_a_named_forced_call_parses(self):
        opening = forced_call({"type": "function", "function": {"name": "read_file"}}, TOOLS, "minimax")
        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, tool_format="minimax")
        events = parser.feed(opening)
        events += parser.feed('<parameter name="path">file.txt</parameter>\n</invoke>\n</minimax:tool_call>')
        events += parser.finish()
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual([(call.name, call.arguments) for call in calls],
                         [("read_file", {"path": "file.txt"})])

    def test_an_any_forced_call_parses(self):
        two = TOOLS + [{"name": "write_file", "parameters": {"properties": {"path": {"type": "string"}}}}]
        opening = forced_call("required", two, "minimax")
        parser = OutputParser(thinking=False, tools=two, stream_tools=True, tool_format="minimax")
        events = parser.feed(opening)
        events += parser.feed('read_file">\n<parameter name="path">file.txt</parameter>\n</invoke>\n'
                              '</minimax:tool_call>')
        events += parser.finish()
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual([(call.name, call.arguments) for call in calls],
                         [("read_file", {"path": "file.txt"})])


class OverHttp(unittest.TestCase):
    """The whole path: Service (the M2 template) -> OutputParser -> OpenAI and Anthropic answers."""

    SCRIPT = "I will read it.\n" + CALL + "\n</think>\n\nDone."

    def test_both_apis_see_the_call(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, self.SCRIPT, max_context=CTX), tok, ChatTemplate(M2_TEMPLATE))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        tools = {
            "openai": [{"type": "function", "function": {"name": "read_file", "description": "read a file",
                                                         "parameters": TOOLS[0]["parameters"]}}],
            "anthropic": [{"name": "read_file", "description": "read a file",
                           "input_schema": TOOLS[0]["parameters"]}],
        }
        expected = {"limit": 120, "path": "file.txt"}
        try:
            for api, path in (("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages")):
                for stream in (False, True):
                    with self.subTest(api=api, stream=stream):
                        body = {"model": "x", "max_tokens": 500, "stream": stream, "tools": tools[api],
                                "messages": [{"role": "user", "content": "read file.txt"}]}
                        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={
                            "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                        with urllib.request.urlopen(req, timeout=30) as response:
                            raw = response.read().decode()
                        if stream:
                            events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                            if api == "openai":
                                name = next(tc["function"]["name"] for event in events
                                            for tc in event["choices"][0]["delta"].get("tool_calls") or []
                                            if tc.get("function", {}).get("name"))
                                args = "".join(tc.get("function", {}).get("arguments") or "" for event in events
                                               for tc in event["choices"][0]["delta"].get("tool_calls") or [])
                                finish = events[-1]["choices"][0]["finish_reason"]
                            else:
                                name = next(event["content_block"]["name"] for event in events
                                            if event["type"] == "content_block_start"
                                            and event["content_block"]["type"] == "tool_use")
                                args = "".join(event["delta"]["partial_json"] for event in events
                                               if event["type"] == "content_block_delta"
                                               and event["delta"]["type"] == "input_json_delta")
                                finish = events[-2]["delta"]["stop_reason"]
                        elif api == "openai":
                            choice = json.loads(raw)["choices"][0]
                            name = choice["message"]["tool_calls"][0]["function"]["name"]
                            args = choice["message"]["tool_calls"][0]["function"]["arguments"]
                            finish = choice["finish_reason"]
                        else:
                            message = json.loads(raw)
                            name = next(block["name"] for block in message["content"] if block["type"] == "tool_use")
                            args = json.dumps(next(block["input"] for block in message["content"]
                                                   if block["type"] == "tool_use"))
                            finish = message["stop_reason"]
                        self.assertEqual(name, "read_file")
                        self.assertEqual(json.loads(args), expected)
                        self.assertEqual(finish, "tool_calls" if api == "openai" else "tool_use")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
