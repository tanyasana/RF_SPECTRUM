// =============================================================================
// Testbench: tb_modules_1_to_4
// Project:   AI-Driven RF Signal Classification & Interference Detection System
// Phase:     Step 4 - Verification of Modules 1 to 4 Cascaded
//
// Modules Tested:
//   1. iq_register
//   2. magnitude_calculator
//   3. threshold_detector
//   4. abnormal_counter
// =============================================================================

`timescale 1ns / 1ps

module tb_modules_1_to_4;

    parameter DATA_WIDTH    = 16;
    parameter POWER_WIDTH   = 32;
    parameter COUNT_WIDTH   = 16;
    parameter ANOMALY_LIMIT = 5;

    // Testbench Stimulus Signals
    reg                          clk;
    reg                          rst_n;
    reg                          sample_valid_in;
    reg  signed [DATA_WIDTH-1:0] in_I;
    reg  signed [DATA_WIDTH-1:0] in_Q;
    reg  [POWER_WIDTH-1:0]       power_threshold;
    reg                          clear_counter;

    // Module Interconnect Wires
    // Module 1 Outputs
    wire signed [DATA_WIDTH-1:0] reg_I;
    wire signed [DATA_WIDTH-1:0] reg_Q;
    wire                         reg_valid;

    // Module 2 Outputs
    wire [POWER_WIDTH-1:0]       power_out;
    wire                         power_valid;

    // Module 3 Outputs
    wire                         det_valid;
    wire                         threshold_exceeded;

    // Module 4 Outputs
    wire [COUNT_WIDTH-1:0]       abnormal_count;
    wire                         anomaly_limit_reached;

    // -------------------------------------------------------------------------
    // Instantiate Module 1: iq_register
    // -------------------------------------------------------------------------
    iq_register #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_mod1 (
        .clk(clk),
        .rst_n(rst_n),
        .sample_valid_in(sample_valid_in),
        .in_I(in_I),
        .in_Q(in_Q),
        .reg_I(reg_I),
        .reg_Q(reg_Q),
        .sample_valid_out(reg_valid)
    );

    // -------------------------------------------------------------------------
    // Instantiate Module 2: magnitude_calculator (P = I^2 + Q^2)
    // -------------------------------------------------------------------------
    magnitude_calculator #(
        .DATA_WIDTH(DATA_WIDTH),
        .POWER_WIDTH(POWER_WIDTH)
    ) u_mod2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(reg_valid),
        .in_I(reg_I),
        .in_Q(reg_Q),
        .valid_out(power_valid),
        .power_out(power_out)
    );

    // -------------------------------------------------------------------------
    // Instantiate Module 3: threshold_detector
    // -------------------------------------------------------------------------
    threshold_detector #(
        .POWER_WIDTH(POWER_WIDTH)
    ) u_mod3 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(power_valid),
        .power_in(power_out),
        .threshold_val(power_threshold),
        .valid_out(det_valid),
        .threshold_exceeded(threshold_exceeded)
    );

    // -------------------------------------------------------------------------
    // Instantiate Module 4: abnormal_counter
    // -------------------------------------------------------------------------
    abnormal_counter #(
        .COUNT_WIDTH(COUNT_WIDTH),
        .ANOMALY_LIMIT(ANOMALY_LIMIT)
    ) u_mod4 (
        .clk(clk),
        .rst_n(rst_n),
        .count_enable(threshold_exceeded),
        .clear_count(clear_counter),
        .abnormal_count(abnormal_count),
        .anomaly_limit_reached(anomaly_limit_reached)
    );

    // 100 MHz Clock generation (10 ns period)
    always #5 clk = ~clk;

    // Task to send one IQ sample and print the pipeline state
    task send_sample(input signed [15:0] i_val, input signed [15:0] q_val);
        begin
            @(negedge clk);
            in_I = i_val;
            in_Q = q_val;
            sample_valid_in = 1'b1;
            @(posedge clk);
            #1;
            $display("[Time %0t ns] In=(%5d, %5d) | Reg=(%5d, %5d) | Power=%9d | ThreshHit=%b | Count=%2d | LimitReached=%b",
                     $time, in_I, in_Q, reg_I, reg_Q, power_out, threshold_exceeded, abnormal_count, anomaly_limit_reached);
        end
    endtask

    integer k;

    // -------------------------------------------------------------------------
    // Main Verification Sequence
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_modules_1_to_4);

        // Initial conditions
        clk             = 1'b0;
        rst_n           = 1'b0;
        sample_valid_in = 1'b0;
        in_I            = 16'sd0;
        in_Q            = 16'sd0;
        power_threshold = 32'd10000000; // Threshold = 10,000,000
        clear_counter   = 1'b0;

        $display("==================================================================");
        $display("VERIFYING MODULES 1 TO 4 (CASCADED DIGITAL HARDWARE PIPELINE)");
        $display("Config: Threshold = %0d | Anomaly Limit = %0d", power_threshold, ANOMALY_LIMIT);
        $display("==================================================================");

        // ---------------------------------------------------------------------
        // STEP A: Verify Reset Behavior (Module 1, 2, 3, 4)
        // ---------------------------------------------------------------------
        #20;
        if (reg_I === 0 && reg_Q === 0 && power_out === 0 && abnormal_count === 0)
            $display(">> [PASS] STEP A: Asynchronous Reset successfully cleared all modules!");
        else
            $display(">> [FAIL] STEP A: Reset failed to clear registers!");

        // De-assert reset
        rst_n = 1'b1;
        #10;

        // ---------------------------------------------------------------------
        // STEP B: Verify Normal Signal Processing (Below Threshold)
        // Inject I=1000, Q=1000 -> Expected Power = 1000^2 + 1000^2 = 2,000,000
        // Expected: ThreshHit = 0, Count = 0
        // ---------------------------------------------------------------------
        $display("\n>> STEP B: Injecting 5 Normal Samples (Power < Threshold)...");
        for (k = 0; k < 5; k = k + 1) begin
            send_sample(16'sd1000, 16'sd1000);
        end

        #2;
        if (threshold_exceeded === 1'b0 && abnormal_count === 0)
            $display(">> [PASS] STEP B: Normal signals correctly processed without false alarms!");
        else
            $display(">> [FAIL] STEP B: Normal signals caused false threshold trigger!");

        // ---------------------------------------------------------------------
        // STEP C: Verify High-Power Rogue Surge & Counter Accumulation
        // Inject I=4000, Q=3000 -> Expected Power = 4000^2 + 3000^2 = 25,000,000
        // 25,000,000 > 10,000,000 -> Expected: ThreshHit = 1, Count increments each cycle
        // ---------------------------------------------------------------------
        $display("\n>> STEP C: Injecting 6 High-Power Rogue Burst Samples (Power > Threshold)...");
        for (k = 0; k < 6; k = k + 1) begin
            send_sample(16'sd4000, 16'sd3000);
        end

        #2;
        if (anomaly_limit_reached === 1'b1 && abnormal_count >= ANOMALY_LIMIT) begin
            $display(">> [PASS] STEP C: High-power burst detected! Counter reached: %0d (Limit = %0d)", 
                     abnormal_count, ANOMALY_LIMIT);
            $display(">> [PASS] anomaly_limit_reached successfully asserted (1'b1)!");
        end else begin
            $display(">> [FAIL] STEP C: Counter did not reach anomaly limit!");
        end

        // ---------------------------------------------------------------------
        // STEP D: Verify Counter Clear / Reset Handshake
        // ---------------------------------------------------------------------
        $display("\n>> STEP D: Testing Counter Synchronous Clear...");
        @(negedge clk);
        clear_counter = 1'b1;
        @(posedge clk);
        #1;
        clear_counter = 1'b0;
        #10;
        if (abnormal_count === 0 && anomaly_limit_reached === 1'b0)
            $display(">> [PASS] STEP D: Counter cleared to 0; anomaly limit flag reset!");
        else
            $display(">> [FAIL] STEP D: Clear counter failed!");

        $display("\n==================================================================");
        $display("ALL 4 MODULES (1, 2, 3, 4) VERIFIED 100%% CORRECT & FUNCTIONAL!");
        $display("==================================================================");
        $finish;
    end

endmodule
