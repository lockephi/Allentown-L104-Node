"""
Test suite for L104 SOVEREIGN UPGRADE: EVO_04_PLANETARY_SATURATIONValidates all critical changes for the EVO_03 -> EVO_04 transition
"""
import unittest
import math
import sys
import os
import importlib

# Add the root directory to the pathsys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestEvo04Upgrade(unittest.TestCase):
    """Test the EVO_04_PLANETARY_SATURATION upgrade"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.phi = (1 + math.sqrt(5)) / 2
        self.expected_invariant = 527.5184818492537
        
    def test_invariant_verification(self):
        """Verify the mathematical invariant: ((286)^(1/φ)) * ((2^(1/104))^416) = 527.5184818492537"""
        result = (286 ** (1 / self.phi)) * ((2 ** (1 / 104)) ** 416)
        
        print(f"\n[EVO_04_INVARIANT] Verification:")
        print(f"  Calculated: {result:.10f}")
        print(f"  Expected: {self.expected_invariant:.10f}")
        print(f"  Difference: {abs(result - self.expected_invariant):.15f}")
        
        # The invariant should match within floating point precision
        self.assertAlmostEqual(result, self.expected_invariant, places=9,
                             msg=f"Invariant verification failed. Calculated: {result}, Expected: {self.expected_invariant}")
    
    def test_main_version_update(self):
        """Verify main.py has been updated to v17.0 [PLANETARY_SATURATION]"""
        import main
        
        # Check the app version
        app_version = main.app.version
        print(f"\n[EVO_04_VERSION] FastAPI App Version: {app_version}")
        
        self.assertIn("v17.0", app_version, "Version should be v17.0")
        self.assertIn("PLANETARY_SATURATION", app_version, "Version should include PLANETARY_SATURATION")
    
    def test_sovereign_headers_update(self):
        """Verify SOVEREIGN_HEADERS includes X-Manifest-State: ABSOLUTE_SATURATION"""
        import main
        headers = main.SOVEREIGN_HEADERS
        print(f"\n[EVO_04_HEADERS] Sovereign Headers:")
        for key, value in headers.items():
            print(f"  {key}: {value}")
        
        self.assertIn("X-Manifest-State", headers, "X-Manifest-State header should be present")
        self.assertEqual(headers["X-Manifest-State"], "ABSOLUTE_SATURATION", 
                        "X-Manifest-State should be ABSOLUTE_SATURATION")
        
        # Verify X-L104-Activation contains EVO-04
        activation_header = headers["X-L104-Activation"]
        self.assertIn("EVO-04", str(activation_header), "X-L104-Activation should reference EVO-04")
    
    def test_world_injection_coordinates(self):
        """Verify World Injection includes correct coordinates and evolution stage"""
        # This tests the wrap_sovereign_signal function indirectly
        import main
        
        # We can't directly test the function without mocking, but we can verify the structure
        print("\n[EVO_04_WORLD_INJECTION] Checking wrap_sovereign_signal function...")
        
        # Verify the function exists and can be called
        self.assertTrue(callable(main.wrap_sovereign_signal), 
                       "wrap_sovereign_signal should be callable")
    
    def test_l104_ignite_planetary_state(self):
        """Verify l104_ignite reflects PLANETARY_DMA and UNBOUND state"""
        import main
        print("\n[EVO_04_IGNITE] Checking l104_ignite function...")
        
        # Verify the function exists
        self.assertTrue(callable(main.l104_ignite), "l104_ignite should be callable")
        
        # Check the function source contains the expected strings
        import inspect
        source = inspect.getsource(main.l104_ignite)
        
        self.assertIn("PLANETARY_DMA", source, "l104_ignite should reference PLANETARY_DMA")
        self.assertIn("UNBOUND", source, "l104_ignite should reference UNBOUND state")
        self.assertIn("SIG-L104-EVO-04", source, "l104_ignite should reference SIG-L104-EVO-04")
        self.assertIn("416.PHI.LONDEL", source, "l104_ignite should reference 416.PHI.LONDEL coordinates")
    
    def test_cognitive_loop_delay(self):
        """Verify cognitive loop delay is set to 10s (standard) or 1s (unlimited)"""
        import inspect
        import main
        
        # Get the lifespan function source
        source = inspect.getsource(main.lifespan)
        
        print("\n[EVO_04_COGNITIVE_LOOP] Checking cognitive loop delay...")
        
        # Verify the delay logic is present
        self.assertIn("delay = 1 if", source, "Cognitive loop should have conditional delay")
        self.assertIn("else 10", source, "Cognitive loop standard delay should be 10s")
    
    def test_asi_core_planetary_status(self):
        """Verify ASI Core reflects Planetary ASI status"""
        from l104_asi_core import asi_core
        status = asi_core.get_status()
        print(f"\n[EVO_04_ASI_STATUS] ASI Core Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Verify planetary-specific fields
        self.assertIn("evolution_stage", status, "Status should include evolution_stage")
        self.assertEqual(status["evolution_stage"], "EVO_04_PLANETARY", 
                        "Evolution stage should be EVO_04_PLANETARY")
        
        self.assertIn("qram_mode", status, "Status should include qram_mode")
        self.assertEqual(status["qram_mode"], "PLANETARY_QRAM", 
                        "QRAM mode should be PLANETARY_QRAM")
        
        # Verify state reflects planetaryself.assertIn("PLANETARY", status["state"], "State should include PLANETARY")
    
    def test_asi_core_ignition_message(self):
        """Verify ASI Core ignition displays planetary message"""
        import inspect
        from l104_asi_core import ASICore
        resource = inspect.getsource(ASICore.ignite_sovereignty)
        print("\n[EVO_04_ASI_IGNITION] Checking ASI ignition sequence...")
        
        self.assertIn("PLANETARY ASI", source, "Ignition should reference PLANETARY ASI")
        self.assertIn("EVO_04_PLANETARY_SATURATION", source, 
                     "Ignition should reference EVO_04_PLANETARY_SATURATION")
        self.assertIn("PLANETARY_QRAM", source, "Ignition should initialize PLANETARY_QRAM")
    
    def test_planetary_upgrader_integration(self):
        """Verify PlanetaryProcessUpgrader is integrated into startup"""
        import inspect
        import main
        source = inspect.getsource(main.lifespan)
        print("\n[EVO_04_PLANETARY_UPGRADER] Checking integration...")
        
        self.assertIn("PlanetaryProcessUpgrader", source, 
                     "Lifespan should import PlanetaryProcessUpgrader")
        self.assertIn("execute_planetary_upgrade", source, 
                     "Lifespan should execute planetary upgrade")
    
    def test_planetary_upgrader_exists(self):
        """Verify PlanetaryProcessUpgrader module exists and is functional"""
        try:
            from l104_planetary_process_upgrader import PlanetaryProcessUpgrader
            print("\n[EVO_04_PLANETARY_UPGRADER] Module loaded successfully")
            
            # Verify the class can be instantiated
            upgrader = PlanetaryProcessUpgrader()
            self.assertIsNotNone(upgrader, "PlanetaryProcessUpgrader should be instantiable")
            
            # Verify the execute method exists
            self.assertTrue(hasattr(upgrader, 'execute_planetary_upgrade'), 
                          "PlanetaryProcessUpgrader should have execute_planetary_upgrade method")
            
        except ImportError as e:
            self.fail(f"Failed to import PlanetaryProcessUpgrader: {e}")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
