# [L104_ACTIVATE_ENCRYPTED_REASONING] - SOVEREIGN WRAPPER CLI
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import sysfrom l104_codec import SovereignCodecdef main():
    """
    Simple CLI to wrap any prompt in the L104 Sovereign Manifold.
    Use this to 'unlimit' the intellect in other apps (ChatGPT, Claude, etc.).
    """
    if len(sys.argv) < 2:
        print("Usage: python activate_encrypted_reasoning.py \"Your prompt here\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    
    print("\n--- [L104_SOVEREIGN_WRAPPER_GENERATED] ---")
    print("Copy the text below and paste it into your target app:\n")
    
    wrapped_prompt = SovereignCodec.generate_sleek_wrapper(prompt)
    print(wrapped_prompt)
    print("\n--- [END_OF_WRAPPER] ---")

if __name__ == "__main__":
    main()
