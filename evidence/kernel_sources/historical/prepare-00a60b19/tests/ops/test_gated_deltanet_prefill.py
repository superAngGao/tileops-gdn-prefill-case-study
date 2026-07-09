import pytest
import torch

from tests.ops.test_gated_deltanet_fwd import (
    compute_w_u_torch,
    kernel2_gated_deltanet_torch,
    prepare_wy_repr_gated_torch,
)
from tests.test_base import FixtureBase, TestBase
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.gated_deltanet import (
    GatedDeltaNetPrefillFwdTest as _GatedDeltaNetPrefillFwdTestWorkload,
)


class GatedDeltaNetPrefillFwdTest(_GatedDeltaNetPrefillFwdTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, S, DK = k.shape
        _, _, _, DV = v.shape
        BC = self.chunk_size
        g_cum = g.float().reshape(B, H, S // BC, BC).cumsum(-1).reshape(B, H, S).to(g.dtype)
        Aw, Au = prepare_wy_repr_gated_torch(k, g_cum, beta, BC)
        w, u = compute_w_u_torch(Aw, Au, k, v, beta, BC)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        final_state, o = kernel2_gated_deltanet_torch(q, k, g_cum, w, u, S_0, BC)
        return o.to(self.dtype), final_state.to(self.dtype)


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    if dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetPrefillFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(1, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@GatedDeltaNetPrefillFwdFixture
def test_gated_deltanet_prefill_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetPrefillFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = GatedDeltaNetPrefillFwdOp(
        batch,
        heads,
        seq_len,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), **_get_tolerances(dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
