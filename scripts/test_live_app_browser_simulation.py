"""
=============================================================================
Module: test_live_app_browser_simulation.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 7 - Automated Browser Simulation & Exception Catching via AppTest
=============================================================================
"""

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from streamlit.testing.v1 import AppTest

def test_streamlit_app():
    print("=" * 80)
    print("RUNNING LIVE STREAMLIT APP TEST VIA APPTEST BROWSER SIMULATION")
    print("=" * 80)
    
    app_path = os.path.join(base_dir, "dashboard", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    
    # 1. Initial launch
    print("\n[TEST 1] Initial App Launch (Default Scenario: Clean QPSK)...")
    at.run()
    
    if at.exception:
        print("  [ERROR] Initial launch raised exception:")
        for exc in at.exception:
            print("   ", exc.value)
        return False
    print("  [PASS] Initial launch succeeded with 0 exceptions!")

    # Find scenario selectbox
    scenario_sb = None
    for sb in at.sidebar.selectbox:
        if "Noisy BPSK (SNR = 6 dB)" in sb.options:
            scenario_sb = sb
            break

    if scenario_sb is None:
        print("  [ERROR] Could not find scenario selectbox in sidebar!")
        return False

    # 2. Noisy BPSK
    print("\n[TEST 2] Switching Scenario to 'Noisy BPSK (SNR = 6 dB)'...")
    scenario_sb.select("Noisy BPSK (SNR = 6 dB)").run()
    if at.exception:
        print("  [ERROR] Noisy BPSK raised exception:")
        for exc in at.exception:
            print("   ", exc.value)
        return False
    print("  [PASS] Noisy BPSK rendered with 0 exceptions!")

    # 3. Clean 16-QAM
    print("\n[TEST 3] Switching Scenario to 'Clean 16-QAM (Multi-Tier Amplitude)'...")
    scenario_sb.select("Clean 16-QAM (Multi-Tier Amplitude)").run()
    if at.exception:
        print("  [ERROR] Clean 16-QAM raised exception:")
        for exc in at.exception:
            print("   ", exc.value)
        return False
    print("  [PASS] Clean 16-QAM rendered with 0 exceptions!")

    # 4. Pulsed Burst Interference
    print("\n[TEST 4] Switching Scenario to 'Pulsed Burst Interference (Energy Surge)'...")
    scenario_sb.select("Pulsed Burst Interference (Energy Surge)").run()
    if at.exception:
        print("  [ERROR] Burst interference raised exception:")
        for exc in at.exception:
            print("   ", exc.value)
        return False
    print("  [PASS] Burst Interference rendered with 0 exceptions!")

    # 5. Swept Chirp Jammer
    print("\n[TEST 5] Switching Scenario to 'Swept Chirp Jammer (Broadband Threat)'...")
    scenario_sb.select("Swept Chirp Jammer (Broadband Threat)").run()
    if at.exception:
        print("  [ERROR] Swept Chirp raised exception:")
        for exc in at.exception:
            print("   ", exc.value)
        return False
    print("  [PASS] Swept Chirp rendered with 0 exceptions!")

    # 6. Live Interactive Synthesis
    print("\n[TEST 6] Switching Mode to 'Live Interactive Synthesis'...")
    at.sidebar.radio[0].set_value("Live Interactive Synthesis").run()
    if at.exception:
        print("  [ERROR] Live Synthesis raised exception:")
        for exc in at.exception:
            print("   ", exc.value)
        return False
    print("  [PASS] Live Interactive Synthesis rendered with 0 exceptions!")

    # 7. Select 8PSK with Jamming Burst
    synth_sb = None
    for sb in at.sidebar.selectbox:
        if "8PSK" in sb.options:
            synth_sb = sb
            break
    if synth_sb:
        print("\n[TEST 7] In Live Synthesis, selecting '8PSK' with Jamming Burst...")
        synth_sb.select("8PSK")
        if at.sidebar.checkbox:
            at.sidebar.checkbox[0].check()
        at.run()
        if at.exception:
            print("  [ERROR] 8PSK synthesis raised exception:")
            for exc in at.exception:
                print("   ", exc.value)
            return False
        print("  [PASS] 8PSK synthesis with burst rendered with 0 exceptions!")

    # 8. Move Hardware Threshold Slider
    print("\n[TEST 8] Moving Hardware Power Threshold Slider (2.5 -> 4.5)...")
    at.sidebar.slider[0].set_value(4.5).run()
    if at.exception:
        print("  [ERROR] Slider move raised exception:")
        for exc in at.exception:
            print("   ", exc.value)
        return False
    print("  [PASS] Slider update rendered with 0 exceptions!")

    print("\n" + "=" * 80)
    print("ALL 8 STREAMLIT APP BROWSER SIMULATION TESTS PASSED COMPLETELY (0 ERRORS)!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_streamlit_app()
    if not success:
        sys.exit(1)
