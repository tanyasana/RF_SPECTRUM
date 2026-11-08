// =============================================================================
// Module Name: detection_fsm
// Project:     AI-Driven RF Signal Classification & Interference Detection System
// Phase:       Step 4 - Digital System Design (Module 5 of 6)
// Description: 4-State Sequential Finite State Machine (FSM) Controller.
//              States:
//                IDLE    (2'b00) : Standby, waiting for sample stream
//                MONITOR (2'b01) : Normal spectrum monitoring
//                ALERT   (2'b10) : Threat detected (Jammer / Rogue power surge)
//                REPORT  (2'b11) : Latched diagnostics output
// =============================================================================

`timescale 1ns / 1ps

module detection_fsm (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       stream_active,          // 1 = valid sample stream flowing
    input  wire       anomaly_limit_reached,  // 1 = abnormal count >= limit
    input  wire       reset_alert,            // 1 = operator acknowledge / reset
    
    output reg  [1:0] current_state,
    output reg        alert_flag,
    output reg        report_ready,
    output reg        clear_counter
);

    // State Encoding using localparam (prevents macro errors)
    localparam STATE_IDLE    = 2'b00;
    localparam STATE_MONITOR = 2'b01;
    localparam STATE_ALERT   = 2'b10;
    localparam STATE_REPORT  = 2'b11;

    reg [1:0] next_state;

    // -------------------------------------------------------------------------
    // 1. State Register (Sequential)
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= STATE_IDLE;
        end
        else begin
            current_state <= next_state;
        end
    end

    // -------------------------------------------------------------------------
    // 2. Next State Logic (Combinational)
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 3. Registered Moore Outputs (Glitch-free)
    // -------------------------------------------------------------------------
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
                    clear_counter <= 1'b1; // Clear count for next window
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
