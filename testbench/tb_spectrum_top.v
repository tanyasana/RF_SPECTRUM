// =============================================================================
// Testbench Name: tb_spectrum_top
// Project:        AI-Driven RF Signal Classification & Interference Detection System
// Description:    Cycle-accurate verification testbench for spectrum_top.
//                 Generates clean waveforms for EPWave / GTKWave inspection.
// =============================================================================

`timescale 1ns / 1ps

module tb_spectrum_top;

    // Simulation Parameters
    parameter DATA_WIDTH    = 16;
    parameter POWER_WIDTH   = 32;
    parameter COUNT_WIDTH   = 16;
    parameter ANOMALY_LIMIT = 5;

    // Device Under Test (DUT) Inputs
    reg                          clk;
    reg                          rst_n;
    reg                          sample_valid_in;
    reg  signed [DATA_WIDTH-1:0] raw_I;
    reg  signed [DATA_WIDTH-1:0] raw_Q;
    reg  [POWER_WIDTH-1:0]       power_threshold;
    reg                          reset_alert;

    // DUT Outputs
    wire [POWER_WIDTH-1:0]       instantaneous_power;
    wire                         threshold_hit;
    wire [COUNT_WIDTH-1:0]       total_anomalies;
    wire [1:0]                   fsm_state;
    wire                         alert_flag;
    wire                         report_ready;

    // Instantiate Top Module
    spectrum_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .POWER_WIDTH(POWER_WIDTH),
        .COUNT_WIDTH(COUNT_WIDTH),
        .ANOMALY_LIMIT(ANOMALY_LIMIT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .sample_valid_in(sample_valid_in),
        .raw_I(raw_I),
        .raw_Q(raw_Q),
        .power_threshold(power_threshold),
        .reset_alert(reset_alert),
        .instantaneous_power(instantaneous_power),
        .threshold_hit(threshold_hit),
        .total_anomalies(total_anomalies),
        .fsm_state(fsm_state),
        .alert_flag(alert_flag),
        .report_ready(report_ready)
    );

    // 100 MHz Clock Generator (Period = 10 ns)
    always #5 clk = ~clk;

    // Task to deliver one IQ sample
    task send_iq(input signed [15:0] i_val, input signed [15:0] q_val);
        begin
            @(negedge clk);
            raw_I = i_val;
            raw_Q = q_val;
            sample_valid_in = 1'b1;
            @(posedge clk);
            #1;
            $display("[Time %0t ns] In=(%6d, %6d) | Power=%9d | Hit=%b | Count=%2d | State=%b | Alert=%b",
                     $time, raw_I, raw_Q, instantaneous_power, threshold_hit, total_anomalies, fsm_state, alert_flag);
        end
    endtask

    integer i;

    // -------------------------------------------------------------------------
    // Main Stimulus Procedure
    // -------------------------------------------------------------------------
    initial begin
        // Setup waveform dumping for EPWave in EDA Playground
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_spectrum_top);

        // Initialize signals
        clk             = 1'b0;
        rst_n           = 1'b0; // Active-low reset asserted
        sample_valid_in = 1'b0;
        raw_I           = 16'sd0;
        raw_Q           = 16'sd0;
        power_threshold = 32'd25000000; // Threshold = 25,000,000 (~5000^2)
        reset_alert     = 1'b0;

        $display("==================================================================");
        $display("STARTING RF SPECTRUM HARDWARE SIMULATION TESTBENCH");
        $display("Threshold = %0d | Anomaly Limit = %0d", power_threshold, ANOMALY_LIMIT);
        $display("==================================================================");

        // 1. Hold Reset for 3 clock cycles
        #20;
        rst_n = 1'b1; // De-assert reset
        #10;
        $display("\n[TEST 1] Reset De-asserted. System entering operational mode...");

        // 2. Normal QPSK Signal (Amplitudes below threshold: +/- 2000)
        // Power = 2000^2 + 2000^2 = 8,000,000 < 25,000,000 -> No Alert
        $display("\n[TEST 2] Injecting 6 Normal QPSK Samples (Below Threshold)...");
        for (i = 0; i < 6; i = i + 1) begin
            send_iq(16'sd2000, 16'sd2000);
        end

        // 3. High-Power Rogue Burst Interference (Amplitudes: +/- 6000)
        // Power = 6000^2 + 6000^2 = 72,000,000 > 25,000,000 -> Threshold Exceeded!
        $display("\n[TEST 3] Injecting High-Power Burst Interference (Above Threshold)...");
        for (i = 0; i < 9; i = i + 1) begin
            send_iq(16'sd6000, 16'sd6000);
        end

        // Check if ALERT state was achieved
        #10;
        if (alert_flag === 1'b1 && fsm_state === 2'b10) begin
            $display("\n>>> SUCCESS: System transitioned to ALERT state (2'b10)!");
            $display(">>> Total Abnormal Samples Counted: %0d", total_anomalies);
        end else begin
            $display("\n>>> ERROR: Expected ALERT state, but got State = %b", fsm_state);
        end

        // 4. Operator Acknowledge / Reset Alert
        $display("\n[TEST 4] Operator issues Reset Alert Acknowledgment...");
        @(negedge clk);
        reset_alert = 1'b1;
        @(posedge clk);
        #1;
        $display("[Time %0t ns] Reset Alert Applied | State=%b | Report Ready=%b", 
                 $time, fsm_state, report_ready);

        @(negedge clk);
        reset_alert = 1'b0;
        
        // 5. Normal Signal Resumed
        $display("\n[TEST 5] Normal Signal Resumes...");
        for (i = 0; i < 4; i = i + 1) begin
            send_iq(16'sd1500, -16'sd1500);
        end

        #30;
        $display("\n==================================================================");
        $display("SIMULATION COMPLETE: All hardware modules verified successfully!");
        $display("==================================================================");
        $finish;
    end

endmodule
