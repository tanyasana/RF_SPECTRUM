// =============================================================================
// EDA PLAYGROUND: DESIGN.SV
// Project: AI-Driven RF Signal Classification & Interference Detection System
// Contains all 5 synthesizable hardware modules + top-level SoC wrapper:
//   1. iq_register
//   2. magnitude_calculator
//   3. threshold_detector
//   4. abnormal_counter
//   5. detection_fsm
//   6. spectrum_top
// =============================================================================

`timescale 1ns / 1ps

// -----------------------------------------------------------------------------
// Module 1: iq_register
// -----------------------------------------------------------------------------
module iq_register #(
    parameter DATA_WIDTH = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          sample_valid_in,
    input  wire signed [DATA_WIDTH-1:0]  in_I,
    input  wire signed [DATA_WIDTH-1:0]  in_Q,
    output reg  signed [DATA_WIDTH-1:0]  reg_I,
    output reg  signed [DATA_WIDTH-1:0]  reg_Q,
    output reg                           sample_valid_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_I            <= {DATA_WIDTH{1'b0}};
            reg_Q            <= {DATA_WIDTH{1'b0}};
            sample_valid_out <= 1'b0;
        end 
        else begin
            if (sample_valid_in) begin
                reg_I            <= in_I;
                reg_Q            <= in_Q;
                sample_valid_out <= 1'b1;
            end 
            else begin
                reg_I            <= reg_I;
                reg_Q            <= reg_Q;
                sample_valid_out <= 1'b0;
            end
        end
    end
endmodule


// -----------------------------------------------------------------------------
// Module 2: magnitude_calculator (Power P = I^2 + Q^2)
// -----------------------------------------------------------------------------
module magnitude_calculator #(
    parameter DATA_WIDTH  = 16,
    parameter POWER_WIDTH = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          valid_in,
    input  wire signed [DATA_WIDTH-1:0]  in_I,
    input  wire signed [DATA_WIDTH-1:0]  in_Q,
    output reg                           valid_out,
    output reg         [POWER_WIDTH-1:0] power_out
);
    wire signed [2*DATA_WIDTH-1:0] i_mult = in_I * in_I;
    wire signed [2*DATA_WIDTH-1:0] q_mult = in_Q * in_Q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            power_out <= {POWER_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end
        else begin
            valid_out <= valid_in;
            if (valid_in)
                power_out <= i_mult[POWER_WIDTH-1:0] + q_mult[POWER_WIDTH-1:0];
            else
                power_out <= {POWER_WIDTH{1'b0}};
        end
    end
endmodule


// -----------------------------------------------------------------------------
// Module 3: threshold_detector
// -----------------------------------------------------------------------------
module threshold_detector #(
    parameter POWER_WIDTH = 32
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   valid_in,
    input  wire [POWER_WIDTH-1:0] power_in,
    input  wire [POWER_WIDTH-1:0] threshold_val,
    output reg                    valid_out,
    output reg                    threshold_exceeded
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            threshold_exceeded <= 1'b0;
            valid_out          <= 1'b0;
        end
        else begin
            valid_out <= valid_in;
            if (valid_in) begin
                if (power_in > threshold_val)
                    threshold_exceeded <= 1'b1;
                else
                    threshold_exceeded <= 1'b0;
            end
            else begin
                threshold_exceeded <= 1'b0;
            end
        end
    end
endmodule


// -----------------------------------------------------------------------------
// Module 4: abnormal_counter
// -----------------------------------------------------------------------------
module abnormal_counter #(
    parameter COUNT_WIDTH   = 16,
    parameter ANOMALY_LIMIT = 5
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   count_enable,
    input  wire                   clear_count,
    output reg  [COUNT_WIDTH-1:0] abnormal_count,
    output wire                   anomaly_limit_reached
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            abnormal_count <= {COUNT_WIDTH{1'b0}};
        end
        else begin
            if (clear_count) begin
                abnormal_count <= {COUNT_WIDTH{1'b0}};
            end
            else if (count_enable) begin
                abnormal_count <= abnormal_count + 1'b1;
            end
        end
    end

    assign anomaly_limit_reached = (abnormal_count >= ANOMALY_LIMIT);
endmodule


// -----------------------------------------------------------------------------
// Module 5: detection_fsm
// -----------------------------------------------------------------------------
module detection_fsm (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       stream_active,
    input  wire       anomaly_limit_reached,
    input  wire       reset_alert,
    output reg  [1:0] current_state,
    output reg        alert_flag,
    output reg        report_ready,
    output reg        clear_counter
);
    localparam STATE_IDLE    = 2'b00;
    localparam STATE_MONITOR = 2'b01;
    localparam STATE_ALERT   = 2'b10;
    localparam STATE_REPORT  = 2'b11;

    reg [1:0] next_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            current_state <= STATE_IDLE;
        else
            current_state <= next_state;
    end

    always @(*) begin
        next_state = current_state;
        case (current_state)
            STATE_IDLE: begin
                if (stream_active)
                    next_state = STATE_MONITOR;
            end
            STATE_MONITOR: begin
                if (!stream_active)
                    next_state = STATE_IDLE;
                else if (anomaly_limit_reached)
                    next_state = STATE_ALERT;
            end
            STATE_ALERT: begin
                if (reset_alert)
                    next_state = STATE_REPORT;
            end
            STATE_REPORT: begin
                if (stream_active)
                    next_state = STATE_MONITOR;
                else
                    next_state = STATE_IDLE;
            end
            default: next_state = STATE_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alert_flag    <= 1'b0;
            report_ready  <= 1'b0;
            clear_counter <= 1'b1;
        end
        else begin
            case (next_state)
                STATE_IDLE: begin
                    alert_flag    <= 1'b0;
                    report_ready  <= 1'b0;
                    clear_counter <= 1'b1;
                end
                STATE_MONITOR: begin
                    alert_flag    <= 1'b0;
                    report_ready  <= 1'b0;
                    clear_counter <= 1'b0;
                end
                STATE_ALERT: begin
                    alert_flag    <= 1'b1;
                    report_ready  <= 1'b0;
                    clear_counter <= 1'b0;
                end
                STATE_REPORT: begin
                    alert_flag    <= 1'b0;
                    report_ready  <= 1'b1;
                    clear_counter <= 1'b1;
                end
                default: begin
                    alert_flag    <= 1'b0;
                    report_ready  <= 1'b0;
                    clear_counter <= 1'b1;
                end
            endcase
        end
    end
endmodule


// -----------------------------------------------------------------------------
// Module 6: spectrum_top (Integrated SoC Wrapper)
// -----------------------------------------------------------------------------
module spectrum_top #(
    parameter DATA_WIDTH    = 16,
    parameter POWER_WIDTH   = 32,
    parameter COUNT_WIDTH   = 16,
    parameter ANOMALY_LIMIT = 5
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          sample_valid_in,
    input  wire signed [DATA_WIDTH-1:0]  raw_I,
    input  wire signed [DATA_WIDTH-1:0]  raw_Q,
    input  wire [POWER_WIDTH-1:0]        power_threshold,
    input  wire                          reset_alert,
    output wire [POWER_WIDTH-1:0]        instantaneous_power,
    output wire                          threshold_hit,
    output wire [COUNT_WIDTH-1:0]        total_anomalies,
    output wire [1:0]                    fsm_state,
    output wire                          alert_flag,
    output wire                          report_ready
);
    wire signed [DATA_WIDTH-1:0] reg_I;
    wire signed [DATA_WIDTH-1:0] reg_Q;
    wire                         reg_valid;
    wire [POWER_WIDTH-1:0]       calc_power;
    wire                         calc_valid;
    wire                         det_valid;
    wire                         det_hit;
    wire                         anomaly_limit_reached;
    wire                         clear_counter_cmd;

    assign instantaneous_power = calc_power;
    assign threshold_hit       = det_hit;

    iq_register #(.DATA_WIDTH(DATA_WIDTH)) u_iq_reg (
        .clk(clk),
        .rst_n(rst_n),
        .sample_valid_in(sample_valid_in),
        .in_I(raw_I),
        .in_Q(raw_Q),
        .reg_I(reg_I),
        .reg_Q(reg_Q),
        .sample_valid_out(reg_valid)
    );

    magnitude_calculator #(.DATA_WIDTH(DATA_WIDTH), .POWER_WIDTH(POWER_WIDTH)) u_mag_calc (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(reg_valid),
        .in_I(reg_I),
        .in_Q(reg_Q),
        .valid_out(calc_valid),
        .power_out(calc_power)
    );

    threshold_detector #(.POWER_WIDTH(POWER_WIDTH)) u_thresh_det (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(calc_valid),
        .power_in(calc_power),
        .threshold_val(power_threshold),
        .valid_out(det_valid),
        .threshold_exceeded(det_hit)
    );

    abnormal_counter #(.COUNT_WIDTH(COUNT_WIDTH), .ANOMALY_LIMIT(ANOMALY_LIMIT)) u_abnormal_cnt (
        .clk(clk),
        .rst_n(rst_n),
        .count_enable(det_hit),
        .clear_count(clear_counter_cmd),
        .abnormal_count(total_anomalies),
        .anomaly_limit_reached(anomaly_limit_reached)
    );

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
