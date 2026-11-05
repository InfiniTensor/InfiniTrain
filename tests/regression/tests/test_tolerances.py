"""Policies are data-extensible while old numerical behavior remains unchanged."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import checks
import tolerances


class TolerancePolicyTests(unittest.TestCase):
    def setUp(self):
        self.registry = copy.deepcopy(tolerances.load_registry())

    def test_shipped_defaults_preserve_all_legacy_thresholds(self):
        for policy in ("distributed_equivalence","oracle"):
            for dtype,(atol,rtol) in checks.TOLERANCES.items():
                for metric in tolerances.METRICS:
                    for backend in ("cpu","cuda","privateuse1","privateuseone"):
                        with self.subTest(policy=policy,dtype=dtype,metric=metric,backend=backend):
                            expected={"atol":0,"rtol":0} if metric=="consumed_samples" else {"atol":atol,"rtol":rtol}
                            self.assertEqual(tolerances.resolve(policy,dtype,metric,backend),expected)
                            self.assertEqual(tolerances.resolve_historical_default(policy,dtype,metric,backend),expected)
        for dtype in tolerances.DTYPES:
            self.assertEqual(tolerances.resolve("lr_schedule",dtype,"lr","cuda"),{"atol":1e-9,"rtol":1e-5})

    def test_new_policy_is_only_a_json_change(self):
        policy=copy.deepcopy(self.registry["policies"]["oracle"])
        policy["metrics"]={"gradient":{"float32":{"atol":2e-7,"rtol":3e-6}}}
        self.registry["policies"]["reduction_v2"]=policy
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"tolerances.json";path.write_text(json.dumps(self.registry))
            self.assertEqual(tolerances.resolve("reduction_v2","float32","gradient","cuda",path),{"atol":2e-7,"rtol":3e-6})
            self.assertEqual(tolerances.resolve("reduction_v2","float32","parameter","cuda",path),{"atol":1e-5,"rtol":1e-5})

    def test_backend_metric_override_does_not_widen_other_metrics(self):
        self.registry["policies"]["oracle"]["backends"]={"cuda":{"gradient":{"bfloat16":{"atol":.02,"rtol":.03}}}}
        resolve=lambda dtype,metric,backend:tolerances.resolve("oracle",dtype,metric,backend,self.registry)
        self.assertEqual(resolve("bfloat16","gradient","cuda"),{"atol":.02,"rtol":.03})
        self.assertEqual(resolve("bfloat16","parameter","cuda"),{"atol":.01,"rtol":.01})
        self.assertEqual(resolve("bfloat16","gradient","cpu"),{"atol":.01,"rtol":.01})
        self.assertEqual(resolve("float32","gradient","cuda"),{"atol":1e-5,"rtol":1e-5})

    def test_backend_alias_has_the_same_explicit_override(self):
        self.registry["policies"]["oracle"]["backends"]={"privateuse1":{"loss":{"float32":{"atol":0,"rtol":0}}}}
        self.assertEqual(tolerances.resolve("oracle","fp32","loss","privateuseone",self.registry),{"atol":0,"rtol":0})

    def test_generic_metric_override_and_historical_defaults_are_independent(self):
        self.registry["policies"]["oracle"]["metrics"]={"loss":{"float32":{"atol":.001,"rtol":.002}}}
        for backend in self.registry["backends"]:
            self.assertEqual(tolerances.resolve("oracle","float32","loss",backend,self.registry),{"atol":.001,"rtol":.002})
        old=tolerances.resolve_historical_default("oracle","float32","loss","cpu")
        self.assertEqual(old,{"atol":1e-5,"rtol":1e-5})
        old["atol"]=99
        self.assertEqual(tolerances.resolve_historical_default("oracle","float32","loss","cpu")["atol"],1e-5)

    def test_integer_metrics_cannot_be_relaxed(self):
        self.registry["policies"]["oracle"]["defaults"]["float32"]={"atol":1,"rtol":1}
        self.assertEqual(tolerances.resolve("oracle","float32","consumed_samples","cuda",self.registry),{"atol":0,"rtol":0})
        self.registry["policies"]["oracle"]["metrics"]={"consumed_samples":{"float32":{"atol":1,"rtol":0}}}
        with self.assertRaisesRegex(ValueError,"exact integer"):
            tolerances.load_registry(self.registry)

    def test_unknown_names_fail_closed(self):
        for arguments in (("missing","float32","loss","cpu"),("oracle","float16","loss","cpu"),
                          ("oracle","float32","los","cpu"),("oracle","float32","loss","cdua"),
                          ([],"float32","loss","cpu")):
            with self.subTest(arguments=arguments),self.assertRaises(ValueError):
                tolerances.resolve(*arguments)

    def test_malformed_policy_thresholds_fail_closed(self):
        for value in (-1,float("nan"),float("inf"),True,"0.01"):
            registry=copy.deepcopy(self.registry)
            registry["policies"]["oracle"]["defaults"]["float32"]["atol"]=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                tolerances.load_registry(registry)
        for patch in ({"inherit":"oracle"},{"metrics":{"los":{"float32":{"atol":0,"rtol":0}}}},
                      {"backends":{"cdua":{"loss":{"float32":{"atol":0,"rtol":0}}}}}):
            registry=copy.deepcopy(self.registry);registry["policies"]["oracle"].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):tolerances.load_registry(registry)

    def test_duplicate_json_keys_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":2,"backends":[],"policies":{}}')
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                tolerances.load_registry(path)


if __name__=="__main__":unittest.main()
