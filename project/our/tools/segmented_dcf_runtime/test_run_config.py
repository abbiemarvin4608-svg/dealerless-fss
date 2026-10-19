import argparse
import unittest

from run import party_affinities, child_environment, channel_offsets, process_resources


class RunConfigurationTests(unittest.TestCase):
    def arguments(self, **updates):
        values = dict(cpus=[10, 12, 14], server_cpus=None, client_cpus=None,
                      cggm_threads=1)
        values.update(updates)
        return argparse.Namespace(**values)

    def test_default_keeps_one_cpu_for_each_party(self):
        self.assertEqual(party_affinities(self.arguments(), {10, 12, 14}), [[10], [12]])

    def test_multithread_run_requires_explicit_sufficient_party_affinities(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            party_affinities(self.arguments(cggm_threads=4), set(range(20)))
        args = self.arguments(cggm_threads=4, server_cpus="0,1,2,3", client_cpus="4,5,6,7")
        self.assertEqual(party_affinities(args, set(range(20))),
                         [[0, 1, 2, 3], [4, 5, 6, 7]])
        args.client_cpus = "4,5"
        with self.assertRaisesRegex(ValueError, "at least"):
            party_affinities(args, set(range(20)))

    def test_rejects_unavailable_or_malformed_affinities(self):
        for value in ("", "0,,1", "0,0", "-1", "20", "1x"):
            with self.assertRaises(ValueError):
                party_affinities(self.arguments(server_cpus=value), set(range(20)))
        for threads in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive"):
                party_affinities(self.arguments(cggm_threads=threads), set(range(20)))

    def test_cggm_worker_limit_is_injected_after_environment_cleanup(self):
        env = child_environment({"SEGFSS_CGGM_THREADS": "99", "SEGFSS_OTHER": "stale",
                                 "SDCF_BACKEND": "stale", "PATH": "/bin"},
                                {"SEGFSS_CGGM_THREADS": "4", "SDCF_BACKEND": "SCI"})
        self.assertEqual(env, {"PATH": "/bin", "SEGFSS_CGGM_THREADS": "4", "SDCF_BACKEND": "SCI"})

    def test_graph_allocation_requires_four_cpus_and_two_participants_per_branch(self):
        args = self.arguments(graph_mode="parallel", activation_workers=2, cggm_threads=2,
                              server_cpus="0,2,4,10", client_cpus="6,8,12,16")
        self.assertEqual(party_affinities(args, set(range(20))), [[0,2,4,10],[6,8,12,16]])
        args.client_cpus = "6,8"
        with self.assertRaisesRegex(ValueError, "at least"):
            party_affinities(args, set(range(20)))
        args.activation_workers = 4
        with self.assertRaisesRegex(ValueError, "at most two"):
            party_affinities(args, set(range(20)))

    def test_graph_channels_reserve_each_owned_protocol_and_do_not_overlap_relay(self):
        channels = channel_offsets(2, "parallel")
        self.assertTrue({300,350,400,450,500,550,600,650,700,820,823,870,920} <= set(channels))
        self.assertFalse(set(channels) & {1000 + value for value in channels})

    def test_inherited_openmp_constraints_are_sanitized(self):
        self.assertEqual(child_environment({"OMP_THREAD_LIMIT":"1", "OMP_DYNAMIC":"TRUE", "PATH":"/bin"},
                                          {"OMP_DYNAMIC":"FALSE"}), {"PATH":"/bin", "OMP_DYNAMIC":"FALSE"})

    def test_offline_overlap_requires_four_cpus_and_at_most_two_cggm_workers(self):
        args = self.arguments(offline_overlap=True, cggm_threads=2,
                              server_cpus="0,2,4,10", client_cpus="6,8,12,16")
        self.assertEqual(party_affinities(args, set(range(20))), [[0,2,4,10],[6,8,12,16]])
        args.client_cpus = "6,8"
        with self.assertRaisesRegex(ValueError, "at least"):
            party_affinities(args, set(range(20)))
        args.client_cpus = "6,8,12,16"
        args.cggm_threads = 4
        with self.assertRaisesRegex(ValueError, "at most two"):
            party_affinities(args, set(range(20)))


class DpfPortAndResourceTests(unittest.TestCase):
    def test_dpf_owned_channels_do_not_overlap_native_or_conversion_channels(self):
        ordinary=set(channel_offsets(2,"parallel"))
        dpf=set(channel_offsets(2,"parallel",True))-ordinary
        expected={2000+150*slot+offset for slot in range(4)
                  for offset in (0,3,50,53,100,103)}
        self.assertEqual(dpf,expected)
        self.assertFalse(dpf & ordinary)
        all_channels=ordinary|dpf
        self.assertFalse(all_channels & {1000+offset for offset in all_channels})

    def test_kernel_resource_snapshot_reports_address_space_high_water(self):
        import os
        values=process_resources(os.getpid())
        self.assertGreaterEqual(values["vm_peak_bytes"],values["vm_size_bytes"])
        self.assertGreater(values["rss_peak_bytes"],0)
        self.assertGreater(values["threads_peak"],0)
        self.assertEqual(process_resources(999999999),{})


if __name__ == "__main__":
    unittest.main()
