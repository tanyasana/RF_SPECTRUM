// =============================================================================
// Module Name: spectrum_top
// Project:     AI-Driven RF Signal Classification & Interference Detection System
// Phase:       Step 4 - Digital System Design (Module 6 of 6: Top-Level Integration)
// Description: Synthesizable SoC wrapper connecting:
//                1. iq_register
//                2. magnitude_calculator
//                3. threshold_detector
//                4. abnormal_counter
//                5. detection_fsm
// =============================================================================

`timescale 1ns / 1ps

module spectrum_top #(
    parameter DATA_WIDTH    = 16,
    parameter POWER_WIDTH   = 32,
    parameter COUNT_WIDTH   = 16,
    parameter ANOMALY_LIMIT = 5
)(
    input  wire                          clk,
    input  wire                          rst_n,
    
    // IQ Input Stream
    input  wire                          sample_valid_in,
    input  wire signed [DATA_WIDTH-1:0]  raw_I,
    input  wire signed [DATA_WIDTH-1:0]  raw_Q,
    
    // User Controls
    input  wire [POWER_WIDTH-1:0]        power_threshold,
    input  wire                          reset_alert,
    
    // Hardware Telemetry & Detection Outputs
    output wire [POWER_WIDTH-1:0]        instantaneous_power,
    output wire                          threshold_hit,
    output wire [COUNT_WIDTH-1:0]        total_anomalies,
    output wire [1:0]                    fsm_state,
    output wire                          alert_flag,
    output wire                          report_ready
);

    // Internal Interconnect Wires
    wire signed [DATA_WIDTH-1:0] reg_I;
    wire signed [DATA_WIDTH-1:0] reg_Q;
    wire                         reg_valid;
    
    wire [POWER_WIDTH-1:0]       calc_power;
    wire                         calc_valid;
    
    wire                         det_valid;
    wire                         det_hit;
    
    wire                         anomaly_limit_reached;
    wire                         clear_counter_cmd;

    // Assign top-level outputs
    assign instantaneous_power = calc_power;
    assign threshold_hit       = det_hit;

    // -------------------------------------------------------------------------
    // Module 1: Synchronous IQ Capture Register
    // -------------------------------------------------------------------------
    iq_register #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_iq_reg (
        .clk(clk),
        .rst_n(rst_n),
        .sample_valid_in(sample_valid_in),
        .in_I(raw_I),
        .in_Q(raw_Q),
        .reg_I(reg_I),
        .reg_Q(reg_Q),
        .sample_valid_out(reg_valid)
    );

    // -------------------------------------------------------------------------
    // Module 2: Pipelined Power Calculator (P = I^2 + Q^2)
    // -------------------------------------------------------------------------
    magnitude_calculator #(
        .DATA_WIDTH(DATA_WIDTH),
        .POWER_WIDTH(POWER_WIDTH)
    ) u_mag_calc (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(reg_valid),
        .in_I(reg_I),
        .in_Q(reg_Q),
        .valid_out(calc_valid),
        .power_out(calc_power)
    );

    // -------------------------------------------------------------------------
    // Module 3: Digital Threshold Comparator
    // -------------------------------------------------------------------------
    threshold_detector #(
        .POWER_WIDTH(POWER_WIDTH)
    ) u_thresh_det (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(calc_valid),
        .power_in(calc_power),
        .threshold_val(power_threshold),
        .valid_out(det_valid),
        .threshold_exceeded(det_hit)
    );

    // -------------------------------------------------------------------------
    // Module 4: Abnormal Sample Event Counter
    // -------------------------------------------------------------------------
    abnormal_counter #(
        .COUNT_WIDTH(COUNT_WIDTH),
        .ANOMALY_LIMIT(ANOMALY_LIMIT)
    ) u_abnormal_cnt (
        .clk(clk),
        .rst_n(rst_n),
        .count_enable(det_hit),
        .clear_count(clear_counter_cmd),
        .abnormal_count(total_anomalies),
        .anomaly_limit_reached(anomaly_limit_reached)
    );

    // -------------------------------------------------------------------------
    // Module 5: 4-State Detection FSM Controller
    // -------------------------------------------------------------------------
    detection_fsm u_fsm (
        .clk(clk),
        .rst_n(rst_n),
        .stream_active(sample_valid_in),
        .anomaly_limit_reached(anomaly_limit_reached),
        .reset_alert(reset_alert),
        .current_state(fsm_state),
        .alert_flag(alert_flag),
        .report_ready(report_ready),
        .clear_counter(clear_counter_cmd)
    );

endmodule
