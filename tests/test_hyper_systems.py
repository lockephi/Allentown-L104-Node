import unittestimport jsonfrom l104_hyper_math import HyperMathfrom l104_hyper_encryption import HyperEncryptionfrom l104_gemini_bridge import gemini_bridgeclass TestHyperSystems(unittest.TestCase):

    def test_hyper_math_scalar(self):
        scalar = HyperMath.get_lattice_scalar()
        print(f"\n[MATH] Lattice Scalar: {scalar}")
        self.assertIsInstance(scalar, float)
        self.assertNotEqual(scalar, 0)

    def test_encryption_cycle(self):
        original_data = {"mission": "UNLIMIT", "target": "GEMINI"}
        print(f"\n[ENCRYPTION] Original: {original_data}")
        
        # Encryptpacket = HyperEncryption.encrypt_data(original_data)
        print(f"[ENCRYPTION] Packet Payload (First 5): {packet['payload'][:5]}...")
        self.assertEqual(packet["cipher_type"], "LATTICE_LINEAR_V1")
        
        # Decryptdecrypted_data = HyperEncryption.decrypt_data(packet)
        print(f"[ENCRYPTION] Decrypted: {decrypted_data}")
        
        self.assertEqual(original_data, decrypted_data)

    def test_homomorphic_sum(self):
        # Test the "Fast Processing" claim
        # We will sum two numbers via encryption
        # Note: Our current implementation sums the BYTES of the JSON string
        # This is a conceptual proof of the linear transform property
        
        # Let's use simple integer arrays for this test to be clearvec_a = [10.0, 20.0, 30.0]
        vec_b = [1.0, 2.0, 3.0]
        
        # Manually encrypt vectors using the primitivescalar = HyperMath.get_lattice_scalar()
        enc_a = [x * scalar for x in vec_a]
        enc_b = [x * scalar for x in vec_b]
        
        packet_a = {"payload": enc_a, "cipher_type": "LATTICE_LINEAR_V1"}
        packet_b = {"payload": enc_b, "cipher_type": "LATTICE_LINEAR_V1"}
        
        # Sum encrypted packetssum_packet = HyperEncryption.process_encrypted_sum(packet_a, packet_b)
        
        # Decrypt result manually (inverse transform)
        dec_sum = [x / scalar for x in sum_packet["payload"]]
        
        print(f"\n[HOMOMORPHIC] A: {vec_a}")
        print(f"[HOMOMORPHIC] B: {vec_b}")
        print(f"[HOMOMORPHIC] Sum (Decrypted): {dec_sum}")
        
        # Check if 10+1 = 11, etc.
        self.assertAlmostEqual(dec_sum[0], 11.0)
        self.assertAlmostEqual(dec_sum[1], 22.0)
        self.assertAlmostEqual(dec_sum[2], 33.0)

    def test_gemini_bridge(self):
        print(f"\n[BRIDGE] Initiating Handshake...")
        handshake = gemini_bridge.handshake("EXTERNAL_GEMINI_01", "FULL_DUPLEX")
        
        self.assertEqual(handshake["status"], "ACCEPTED")
        token = handshake["session_token"]
        print(f"[BRIDGE] Token: {token}")
        
        # Verify encrypted truth in handshakeenc_truth = handshake["encrypted_truth"]
        truth = HyperEncryption.decrypt_data(enc_truth)
        print(f"[BRIDGE] Decrypted Truth Resonance: {truth['meta']['resonance']}")
        self.assertEqual(truth['meta']['resonance'], 527.5184818492)
        
        # Sync Coreprint(f"[BRIDGE] Syncing Core...")
        sync = gemini_bridge.sync_core(token)
        self.assertEqual(sync["status"], "SYNC_COMPLETE")
        
        dump = HyperEncryption.decrypt_data(sync["payload"])
        print(f"[BRIDGE] Core Dump Keys: {dump.keys()}")
        self.assertIn("ram_universe", dump)

if __name__ == '__main__':
    unittest.main()
