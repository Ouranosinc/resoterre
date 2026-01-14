import numpy as np
import torch

from resoterre.hybrid_data_loaders import rdps_to_hrdps


def test_post_process_model_output():
    dummy_data = torch.rand(4, 3, 32, 32)  # [0, 1] values
    # Setting temperature min and max
    dummy_data[2, 0, 14, 13] = -1.0
    dummy_data[3, 0, 12, 31] = 1.0
    # Setting precipitation min and max
    dummy_data[0, 1, 5, 7] = -1.0
    dummy_data[1, 1, 21, 20] = 1.0
    # Setting u-component wind min and max
    dummy_data[1, 2, 3, 0] = -1.0
    dummy_data[3, 2, 23, 12] = 1.0
    processed_data = rdps_to_hrdps.post_process_model_output(
        dummy_data, ["HRDPS_P_TT_10000", "HRDPS_P_PR_SFC", "HRDPS_P_UUC_10000"]
    )
    assert "HRDPS_P_TT_10000" in processed_data
    assert "HRDPS_P_PR_SFC" in processed_data
    assert "HRDPS_P_UUC_10000" in processed_data
    assert processed_data["HRDPS_P_TT_10000"].min() == -40.0
    assert processed_data["HRDPS_P_TT_10000"].max() == 40.0
    assert np.isclose(processed_data["HRDPS_P_PR_SFC"].min(), 0.0)
    assert np.isclose(processed_data["HRDPS_P_PR_SFC"].max(), 0.001)
    assert processed_data["HRDPS_P_UUC_10000"].min() == -100.0
    assert processed_data["HRDPS_P_UUC_10000"].max() == 100.0
