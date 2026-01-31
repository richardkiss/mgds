"""
Unit tests for AspectBucketing module to verify enable_resolution_override flag behavior.
"""
import unittest
import torch
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState


class TestAspectBucketingOverride(unittest.TestCase):
    """Test that AspectBucketing respects enable_resolution_override flag."""

    def test_bucket_pool_respects_enable_resolution_override(self):
        """
        Test that start() only includes override resolutions when enable_resolution_override is True.
        
        This test verifies that:
        1. When enable_resolution_override=True, override resolutions are included in bucket pool
        2. When enable_resolution_override=False, override resolutions are NOT included in bucket pool
        3. Global resolutions are always included in bucket pool
        """
        # Create a simple pipeline with AspectBucketing
        input_modules = [
            AspectBucketing(
                quantization=8,
                resolution_in_name='resolution',
                target_resolution_in_name='settings.resolution',
                enable_target_resolutions_override_in_name='concept.enable_resolution_override',
                target_resolutions_override_in_name='concept.resolution_override',
                target_frames_in_name='concept.frames',
                frame_dim_enabled=False,
                scale_resolution_out_name='scale_resolution',
                crop_resolution_out_name='crop_resolution',
                possible_resolutions_out_name='possible_resolutions',
            ),
        ]

        # Create a dataset with:
        # - Global resolution: 1024
        # - Concept A: enable_resolution_override=True, resolution_override="512"
        # - Concept B: enable_resolution_override=True, resolution_override="768"
        # - Concept C: enable_resolution_override=False, resolution_override="512" (should be ignored)
        ds = MGDS(
            device=torch.device('cpu'),
            concepts=[
                {
                    'name': 'ConceptA',
                    'path': '/tmp/fake_path_a',
                    'enable_resolution_override': True,
                    'resolution_override': '512',
                    'frames': 1,
                },
                {
                    'name': 'ConceptB',
                    'path': '/tmp/fake_path_b',
                    'enable_resolution_override': True,
                    'resolution_override': '768',
                    'frames': 1,
                },
                {
                    'name': 'ConceptC',
                    'path': '/tmp/fake_path_c',
                    'enable_resolution_override': False,
                    'resolution_override': '512',  # This should be ignored
                    'frames': 1,
                },
            ],
            settings={
                "resolution": "1024",
            },
            definition=[
                input_modules,
            ],
            batch_size=1,
            state=PipelineState(),
            seed=42,
        )

        # Get the aspect bucketing module from the pipeline
        aspect_bucketing = ds.loading_pipeline.modules[2]  # Skip ConceptPipelineModule and SettingsPipelineModule
        
        # Start the bucketing process (this is where the bug was)
        aspect_bucketing.start(variation=0)
        
        # Verify that bucket_resolutions contains only the enabled resolutions
        # Expected: {512, 768, 1024} - where 512 and 768 come from enabled overrides, 1024 is global
        bucket_resolutions_keys = set(aspect_bucketing.bucket_resolutions.keys())
        
        # Should include:
        # - 1024 (global resolution)
        # - 512 (ConceptA with enable_resolution_override=True)
        # - 768 (ConceptB with enable_resolution_override=True)
        # Should NOT include duplicate 512 from ConceptC (enable_resolution_override=False)
        expected_resolutions = {512, 768, 1024}
        
        self.assertEqual(
            bucket_resolutions_keys,
            expected_resolutions,
            f"Bucket pool should only include resolutions with enable_resolution_override=True. "
            f"Expected: {expected_resolutions}, Got: {bucket_resolutions_keys}"
        )
        
        # Verify that we have buckets created for all expected resolutions
        for resolution in expected_resolutions:
            self.assertIn(
                resolution,
                aspect_bucketing.bucket_resolutions,
                f"Resolution {resolution} should have buckets created"
            )
            self.assertGreater(
                len(aspect_bucketing.bucket_resolutions[resolution]),
                0,
                f"Resolution {resolution} should have at least one bucket"
            )

    def test_bucket_pool_with_all_overrides_disabled(self):
        """
        Test that when all overrides are disabled, only global resolution is used.
        """
        input_modules = [
            AspectBucketing(
                quantization=8,
                resolution_in_name='resolution',
                target_resolution_in_name='settings.resolution',
                enable_target_resolutions_override_in_name='concept.enable_resolution_override',
                target_resolutions_override_in_name='concept.resolution_override',
                target_frames_in_name='concept.frames',
                frame_dim_enabled=False,
                scale_resolution_out_name='scale_resolution',
                crop_resolution_out_name='crop_resolution',
                possible_resolutions_out_name='possible_resolutions',
            ),
        ]

        ds = MGDS(
            device=torch.device('cpu'),
            concepts=[
                {
                    'name': 'ConceptA',
                    'path': '/tmp/fake_path_a',
                    'enable_resolution_override': False,
                    'resolution_override': '512',
                    'frames': 1,
                },
                {
                    'name': 'ConceptB',
                    'path': '/tmp/fake_path_b',
                    'enable_resolution_override': False,
                    'resolution_override': '768',
                    'frames': 1,
                },
            ],
            settings={
                "resolution": "1024",
            },
            definition=[
                input_modules,
            ],
            batch_size=1,
            state=PipelineState(),
            seed=42,
        )

        aspect_bucketing = ds.loading_pipeline.modules[2]
        aspect_bucketing.start(variation=0)
        
        # Should only include global resolution (1024)
        bucket_resolutions_keys = set(aspect_bucketing.bucket_resolutions.keys())
        expected_resolutions = {1024}
        
        self.assertEqual(
            bucket_resolutions_keys,
            expected_resolutions,
            f"With all overrides disabled, bucket pool should only include global resolution. "
            f"Expected: {expected_resolutions}, Got: {bucket_resolutions_keys}"
        )


if __name__ == '__main__':
    unittest.main()

