import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

from wildfire.sim import Config, Game  # noqa: E402
from wildfire.view import View  # noqa: E402


def test_renders_headless_with_overlays(tmp_path):
    game = Game(Config(seed=0, duration=120.0, prm_n=500, prm_candidates=40))
    view = View(game, headless=True)
    view.show_roadmap = view.show_spread = view.show_unreachable = True
    for _ in range(60):
        game.step()
    view.render()
    path = view.screenshot(tmp_path / "frame.png")
    view.close()
    assert path.stat().st_size > 10_000
