import unittest

import numpy as np

from flappy.game import Game


class FlappyGameTests(unittest.TestCase):
    def test_pixels_are_rgb_and_deterministic(self):
        a = Game(seed=7)
        b = Game(seed=7)
        self.assertEqual(a.pixels().shape, (480, 640, 3))
        self.assertEqual(a.pixels().dtype, np.uint8)
        np.testing.assert_array_equal(a.pixels(), b.pixels())

    def test_attack_is_the_only_control(self):
        game = Game(seed=7)
        game.act({"attack": True, "turn": -6, "forward": -20})
        after_flap = game.observation()
        self.assertLess(after_flap["bird_v"], 0)

        comparison = Game(seed=7)
        comparison.act({"attack": False, "turn": 6, "forward": 20})
        self.assertGreater(comparison.observation()["bird_v"], 0)

    def test_no_flap_eventually_ends_episode(self):
        game = Game(seed=7)
        while not game.observation()["finished"]:
            game.act(False)
        self.assertLess(game.observation()["tick"], game.max_ticks)

    def test_reset_is_repeatable_for_episode_number(self):
        a = Game(seed=12)
        b = Game(seed=12)
        a.new_episode()
        b.new_episode()
        np.testing.assert_array_equal(a.pixels(), b.pixels())


if __name__ == "__main__":
    unittest.main()
