from resoterre.ml import ml_loops


def test_minimum_tracker():
    minimum_tracker = ml_loops.MinimumTracker()
    minimum_tracker.update_minimum(iteration=1, value=0.5)
    assert minimum_tracker.value == 0.5
    assert minimum_tracker.iteration == 1
    minimum_tracker.update_minimum(iteration=2, value=0.4)
    assert minimum_tracker.value == 0.4
    assert minimum_tracker.iteration == 2
    minimum_tracker.update_minimum(iteration=3, value=0.6)
    assert minimum_tracker.value == 0.4
    assert minimum_tracker.iteration == 2


def test_minima_tracker():
    minima_tracker = ml_loops.MinimaTracker(minimum_metrics_to_track=["metric1", "metric2"])
    minima_tracker.update_minima(iteration=1, metrics_values={"metric1": 0.5, "metric2": 0.3})
    assert minima_tracker["metric1"].value == 0.5
    assert minima_tracker["metric2"].value == 0.3
    minima_tracker.update_minima(iteration=2, metrics_values={"metric1": 0.4, "metric2": 0.4})
    assert minima_tracker["metric1"].value == 0.4
    assert minima_tracker["metric2"].value == 0.3
    minima_tracker.update_minima(iteration=3, metrics_values={"metric1": 0.6, "metric2": 0.2})
    assert minima_tracker["metric1"].value == 0.4
    assert minima_tracker["metric2"].value == 0.2


def test_loop_state():
    loop_state = ml_loops.LoopState(name="test")
    assert loop_state.name == "test"
    assert loop_state.lifetime_iteration == -1
    assert loop_state.iteration_since_restart == -1
    assert loop_state.iteration_progress == 1.0
    loop_state.start()
    assert loop_state.start_time_lifetime is not None
    assert loop_state.start_time_since_restart is not None
    loop_state.next_iteration()
    assert loop_state.lifetime_iteration == 0
    assert loop_state.iteration_since_restart == 0
    assert loop_state.iteration_progress == 0.0
    loop_state.next_iteration()
    assert loop_state.lifetime_iteration == 1
    assert loop_state.iteration_since_restart == 1
    loop_state.done()
    assert loop_state.end_time is not None
    assert loop_state.stopped_early is False
    assert loop_state.iteration_progress == 1.0
    assert loop_state.nb_done_lifetime() == 2
    assert loop_state.nb_done_since_restart() == 2
