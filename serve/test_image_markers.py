"""Which token stands for a picture's cells, and how a prompt's markers become its rows.

Step-3.7-Flash's chat template writes a bare `<im_patch>` per picture and its vocabulary has no `<|image_pad|>` at all,
so the Qwen-shaped pair test (`<|vision_start|>` then `<|image_pad|>`) matched nothing and every picture came back as
"400 invalid request: the prompt and its images do not match" - before the encoder had even run.

    python -m serve.test_image_markers
"""
import unittest

import serve.server as server

PAD, START, TEXT = 900, 901, 902          # a marker, the marker a picture must follow, and an ordinary id


def ids(*parts):
    out = []
    for p in parts:
        out += list(p) if isinstance(p, (list, tuple)) else [p]
    return out


class Expansion(unittest.TestCase):
    def test_a_family_whose_template_writes_the_marker_alone(self):
        """Step-3.7-Flash: <im_patch> straight in the text, one per picture."""
        prompt = ids(1, 2, PAD, 3, PAD, 4)
        got = server.expand_image_cells(prompt, PAD, None, [3, 2], [TEXT])
        self.assertEqual(got, ids(1, 2, [PAD] * 3, 3, [PAD] * 2, 4))

    def test_the_qwen_shape_still_needs_its_start_marker(self):
        prompt = ids(1, START, PAD, 2, START, PAD, 3)
        got = server.expand_image_cells(prompt, PAD, START, [2, 1], [TEXT])
        self.assertEqual(got, ids(1, START, [PAD] * 2, 2, START, PAD, 3))

    def test_a_marker_without_its_start_is_text(self):
        prompt = ids(1, PAD, 2, START, PAD, 3)
        got = server.expand_image_cells(prompt, PAD, START, [1], [TEXT])
        self.assertEqual(got, ids(1, TEXT, 2, START, PAD, 3))

    def test_one_picture_that_takes_one_cell(self):
        self.assertEqual(server.expand_image_cells(ids(1, PAD, 2), PAD, None, [1], [TEXT]), ids(1, PAD, 2))

    def test_more_pictures_than_markers_is_a_400(self):
        with self.assertRaises(ValueError):
            server.expand_image_cells(ids(1, PAD, 2), PAD, None, [2, 2], [TEXT])

    def test_fewer_pictures_than_markers_is_a_400(self):
        with self.assertRaises(ValueError):
            server.expand_image_cells(ids(1, PAD, 2, PAD, 3), PAD, None, [2], [TEXT])

    def test_no_marker_for_the_pictures_is_a_400(self):
        with self.assertRaises(ValueError):
            server.expand_image_cells(ids(1, 2, 3), PAD, None, [2], [TEXT])


class Markers(unittest.TestCase):
    def test_step35_is_the_bare_marker(self):
        self.assertEqual(server.IMAGE_MARKERS["step35"], ("<im_patch>", None))

    def test_every_other_architecture_keeps_the_qwen_pair(self):
        self.assertEqual(server.IMAGE_MARKERS.get("qwen35moe", (server.IMAGE_PAD, server.VISION_START)),
                         (server.IMAGE_PAD, server.VISION_START))


if __name__ == "__main__":
    unittest.main()
