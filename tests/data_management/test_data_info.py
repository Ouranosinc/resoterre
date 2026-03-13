from resoterre.data_management import data_info


def test_data_info():
    di = data_info.DataInfo(categories={"All"})
    assert di.categories == {"All"}
    di.set_bool("exists", True)
    di.set_min("min", 0, num_values=6)
    di.set_max("max", 1, num_values=6)
    di.set_mean("mean", 0.5, num_values=6)
    di.set_shape("shape", (2, 3))
    di.init_none({"processed": "bool"})
    assert di.bool_info == {"exists": True, "processed": None}
