"""CPU unit tests for MoE combine token-band gating, zero-copy Q reshape, and selective post-MoE AR fusion."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMoeCombineQprepAndSelectiveARFusions(CustomTestCase):
    def test_environ_registration_override_and_export(self):
        from sglang.srt.environ import envs, exportable_env_vars

        expected_fields = {
            "SGLANG_DISABLE_ZERO_COPY_Q": (False, True),
            "SGLANG_FUSED_MOE_SUM_ALL_REDUCE_MIN_TOKENS": (2, 4),
            "SGLANG_FUSED_MOE_SUM_ALL_REDUCE_MAX_TOKENS": (256, 0),
            "SGLANG_DISABLE_POST_MOE_ALLREDUCE_FUSION": (False, True),
        }

        for name, (default_val, override_val) in expected_fields.items():
            field = getattr(envs, name)
            with self.subTest(field=name), patch.dict(os.environ):
                os.environ.pop(name, None)
                self.assertEqual(field.get(), default_val)
                with field.override(override_val):
                    self.assertEqual(field.get(), override_val)
                    self.assertIn(name, exportable_env_vars())
                self.assertEqual(field.get(), default_val)

    def test_reshape_q_zero_copy_and_layout_guards(self):
        try:
            import sgl_kernel  # noqa: F401
        except ImportError:
            stub = types.ModuleType("sgl_kernel")
            stub.merge_state_v2 = None
            sys.modules["sgl_kernel"] = stub
            self.addCleanup(sys.modules.pop, "sgl_kernel", None)

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.flashattention_backend import _reshape_q

        num_heads, head_dim = 16, 128
        q_size = num_heads * head_dim

        # 1. QKV-split non-contiguous 2D tensor: stride=(2560, 1)
        qkv = torch.randn(16, q_size + 256 + 256, dtype=torch.bfloat16)
        q, _, _ = qkv.split([q_size, 256, 256], dim=-1)
        self.assertFalse(q.is_contiguous())
        self.assertEqual(q.stride(), (2560, 1))

        with envs.SGLANG_DISABLE_ZERO_COPY_Q.override(False):
            out = _reshape_q(q, num_heads, head_dim)
            self.assertEqual(out.shape, (16, num_heads, head_dim))
            self.assertEqual(out.stride(), (2560, head_dim, 1))
            self.assertFalse(out.is_contiguous())
            self.assertEqual(out.data_ptr(), q.data_ptr())
            self.assertTrue(
                torch.equal(out, q.contiguous().view(16, num_heads, head_dim))
            )

        # 2. Escape hatch materializes contiguous tensor
        with envs.SGLANG_DISABLE_ZERO_COPY_Q.override(True):
            out_copy = _reshape_q(q, num_heads, head_dim)
            self.assertEqual(out_copy.shape, (16, num_heads, head_dim))
            self.assertTrue(out_copy.is_contiguous())
            self.assertNotEqual(out_copy.data_ptr(), q.data_ptr())
            self.assertTrue(
                torch.equal(out_copy, q.contiguous().view(16, num_heads, head_dim))
            )

        # 3. Already-shaped 3D strided tensor is returned zero-copy
        q3_strided = q.view(16, num_heads, head_dim)
        with envs.SGLANG_DISABLE_ZERO_COPY_Q.override(False):
            out3 = _reshape_q(q3_strided, num_heads, head_dim)
            self.assertIs(out3, q3_strided)

        # 4. Non-mergeable 3D outer-strided input succeeds with unit inner stride
        q_outer_strided = torch.randn(4, 8, q_size, dtype=torch.bfloat16)[:, ::2, :]
        with envs.SGLANG_DISABLE_ZERO_COPY_Q.override(False):
            out_outer = _reshape_q(q_outer_strided, num_heads, head_dim)
            self.assertEqual(out_outer.shape, (16, num_heads, head_dim))
            self.assertEqual(out_outer.stride(-1), 1)
            self.assertTrue(
                torch.equal(
                    out_outer,
                    q_outer_strided.contiguous().view(-1, num_heads, head_dim),
                )
            )

        # 5. Transposed tensor with stride(-1) != 1 falls back to contiguous
        q_transposed = torch.randn(q_size, 16, dtype=torch.bfloat16).t()
        self.assertNotEqual(q_transposed.stride(-1), 1)
        with envs.SGLANG_DISABLE_ZERO_COPY_Q.override(False):
            out_t = _reshape_q(q_transposed, num_heads, head_dim)
            self.assertEqual(out_t.shape, (16, num_heads, head_dim))
            self.assertEqual(out_t.stride(-1), 1)
            self.assertTrue(out_t.is_contiguous())
            self.assertTrue(
                torch.equal(
                    out_t,
                    q_transposed.contiguous().view(-1, num_heads, head_dim),
                )
            )

    def test_fused_moe_sum_all_reduce_guards_and_combine(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

        exec_stub = SimpleNamespace(
            moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        base_kwargs = dict(
            topk=8,
            no_combine=False,
            defer_sum=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            hooks=None,
        )

        with (
            patch.object(fused_moe, "_is_cuda", True),
            patch.object(fused_moe, "_is_musa", False),
            patch.object(fused_moe, "_is_hip", False),
            patch.object(fused_moe, "_use_aiter", False),
            patch.object(fused_moe, "get_exec", return_value=exec_stub),
            patch.object(
                fused_moe, "is_batch_invariant_mode_enabled", return_value=False
            ),
        ):
            # Default OFF
            self.assertFalse(
                fused_moe._should_use_fused_moe_sum_all_reduce(
                    num_tokens=16, **base_kwargs
                )
            )

            # Enabled via --enable-fused-moe-sum-all-reduce with default [2, 256] band
            exec_stub.moe.enable_fused_moe_sum_all_reduce = True
            with (
                envs.SGLANG_FUSED_MOE_SUM_ALL_REDUCE_MIN_TOKENS.override(2),
                envs.SGLANG_FUSED_MOE_SUM_ALL_REDUCE_MAX_TOKENS.override(256),
            ):
                self.assertFalse(
                    fused_moe._should_use_fused_moe_sum_all_reduce(
                        num_tokens=1, **base_kwargs
                    )
                )
                for m in (2, 16, 64, 256):
                    self.assertTrue(
                        fused_moe._should_use_fused_moe_sum_all_reduce(
                            num_tokens=m, **base_kwargs
                        )
                    )
                for m in (257, 1024, 8192):
                    self.assertFalse(
                        fused_moe._should_use_fused_moe_sum_all_reduce(
                            num_tokens=m, **base_kwargs
                        )
                    )

                for override_kw in (
                    {"topk": 2},
                    {"no_combine": True},
                    {"defer_sum": True},
                    {"use_int8_w8a16": True},
                    {"use_int4_w4a16": True},
                    {"hooks": SimpleNamespace(after_down=object())},
                ):
                    self.assertFalse(
                        fused_moe._should_use_fused_moe_sum_all_reduce(
                            num_tokens=16, **{**base_kwargs, **override_kw}
                        )
                    )

            # HIP / ROCm stays supported; CPU / XPU decline
            with (
                envs.SGLANG_FUSED_MOE_SUM_ALL_REDUCE_MIN_TOKENS.override(1),
                envs.SGLANG_FUSED_MOE_SUM_ALL_REDUCE_MAX_TOKENS.override(0),
            ):
                self.assertTrue(
                    fused_moe._should_use_fused_moe_sum_all_reduce(
                        num_tokens=1, **base_kwargs
                    )
                )
                self.assertTrue(
                    fused_moe._should_use_fused_moe_sum_all_reduce(
                        num_tokens=8192, **base_kwargs
                    )
                )

                with patch.object(fused_moe, "_is_cuda", False):
                    self.assertFalse(
                        fused_moe._should_use_fused_moe_sum_all_reduce(
                            num_tokens=16, **base_kwargs
                        )
                    )
                    with patch.object(fused_moe, "_is_hip", True):
                        self.assertTrue(
                            fused_moe._should_use_fused_moe_sum_all_reduce(
                                num_tokens=16, **base_kwargs
                            )
                        )

                exec_stub.deterministic.enable_deterministic_inference = True
                self.assertFalse(
                    fused_moe._should_use_fused_moe_sum_all_reduce(
                        num_tokens=16, **base_kwargs
                    )
                )
                exec_stub.deterministic.enable_deterministic_inference = False

                with patch.object(
                    fused_moe, "is_batch_invariant_mode_enabled", return_value=True
                ):
                    self.assertFalse(
                        fused_moe._should_use_fused_moe_sum_all_reduce(
                            num_tokens=16, **base_kwargs
                        )
                    )

            # moe_combine_topk with intermediate_cache3=None on CUDA vs AITER
            out = torch.full((4, 16), 2.0, dtype=torch.bfloat16)
            res = fused_moe.moe_combine_topk(
                intermediate_cache3=None,
                out_hidden_states=out,
                routed_scaling_factor=2.5,
                use_fused_moe_sum_all_reduce=True,
                out_slice=out,
            )
            self.assertIs(res, out)
            self.assertTrue(torch.all(res == 5.0))

            with patch.object(fused_moe, "_use_aiter", True):
                out_aiter = torch.full((4, 16), 2.0, dtype=torch.bfloat16)
                res_aiter = fused_moe.moe_combine_topk(
                    intermediate_cache3=None,
                    out_hidden_states=out_aiter,
                    routed_scaling_factor=2.5,
                    use_fused_moe_sum_all_reduce=True,
                    out_slice=out_aiter,
                )
                self.assertTrue(torch.all(res_aiter == 2.0))

    def test_selective_post_moe_allreduce_fusion_and_bootstrap(self):
        from sglang.srt.distributed import bootstrap
        from sglang.srt.environ import envs
        from sglang.srt.layers.layer_boundary.contracts import FfnInputFusion
        from sglang.srt.layers.layer_boundary.fusions import allreduce as ar_fusions
        from sglang.srt.layers.layer_boundary.layout import SumGroup
        from sglang.srt.layers.layer_boundary.output import UnreducedOutput

        group_sentinel = object()
        norm_stub = SimpleNamespace(
            forward_with_allreduce_fusion=lambda *args, **kwargs: ("ok", "res"),
        )
        backend_marker = ("backend_fusion",)
        plan_stub = SimpleNamespace(
            norm=norm_stub,
            fusions=SimpleNamespace(
                attn_input_fusions=lambda _plan: (backend_marker,),
                ffn_input_fusions=lambda _plan: (),
            ),
        )
        owed = UnreducedOutput(partial=torch.zeros(16, 2048), group=group_sentinel)
        residual = torch.zeros(16, 2048)

        with (
            patch.object(
                ar_fusions, "post_experts_reduction_group", return_value=group_sentinel
            ),
            patch.object(ar_fusions, "aiter_ar_fusion_applies", return_value=False),
            patch.object(ar_fusions, "flashinfer_ar_fusion_applies", return_value=True),
        ):
            with envs.SGLANG_DISABLE_POST_MOE_ALLREDUCE_FUSION.override(False):
                attn_f = ar_fusions.attn_input_fusions(plan_stub)
                self.assertEqual(len(attn_f), 2)
                self.assertEqual(attn_f[0], backend_marker)
                self.assertEqual(
                    ar_fusions.fused_attn_input(
                        plan_stub,
                        owed,
                        residual,
                        None,
                        None,
                        fuses_quant=False,
                        keep_bf16=False,
                    ),
                    ("ok", "res"),
                )

            with envs.SGLANG_DISABLE_POST_MOE_ALLREDUCE_FUSION.override(True):
                self.assertEqual(
                    ar_fusions.attn_input_fusions(plan_stub), (backend_marker,)
                )
                self.assertIsNone(
                    ar_fusions.fused_attn_input(
                        plan_stub,
                        owed,
                        residual,
                        None,
                        None,
                        fuses_quant=False,
                        keep_bf16=False,
                    )
                )
                ffn_f = ar_fusions.ffn_input_fusions(plan_stub)
                self.assertEqual(len(ffn_f), 1)
                self.assertIsInstance(ffn_f[0], FfnInputFusion)
                self.assertIs(ffn_f[0].completes, SumGroup.ATTN_TP)

        recorded: dict[str, bool] = {}
        comm_exec = SimpleNamespace(
            comm=SimpleNamespace(
                disable_custom_all_reduce=False,
                enable_mscclpp=False,
                enable_torch_symm_mem=False,
                flashinfer_allreduce_fusion_backend="trtllm",
            )
        )
        with (
            patch.object(bootstrap, "get_exec", return_value=comm_exec),
            patch.object(
                bootstrap,
                "set_custom_all_reduce",
                lambda v: recorded.__setitem__("ca", v),
            ),
            patch.object(
                bootstrap, "set_mscclpp", lambda v: recorded.__setitem__("mscclpp", v)
            ),
            patch.object(
                bootstrap,
                "set_torch_symm_mem_all_reduce",
                lambda v: recorded.__setitem__("symm", v),
            ),
            patch.object(
                bootstrap,
                "set_flashinfer_allreduce_only",
                lambda v: recorded.__setitem__("fi_only", v),
            ),
            patch(
                "sglang.srt.layers.flashinfer_comm_fusion.uses_cutedsl_ar_fusion",
                return_value=False,
            ),
        ):
            with envs.SGLANG_DISABLE_POST_MOE_ALLREDUCE_FUSION.override(False):
                bootstrap._set_all_reduce_flags()
                self.assertTrue(recorded["fi_only"])
                self.assertTrue(recorded["ca"])

            with envs.SGLANG_DISABLE_POST_MOE_ALLREDUCE_FUSION.override(True):
                bootstrap._set_all_reduce_flags()
                self.assertFalse(recorded["fi_only"])
                self.assertTrue(recorded["ca"])


if __name__ == "__main__":
    unittest.main()
