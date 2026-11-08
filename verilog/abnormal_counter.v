// =============================================================================
// Module Name: abnormal_counter
// Project:     AI-Driven RF Signal Classification & Interference Detection System
// Phase:       Step 4 - Digital System Design (Module 4 of 6)
// Description: Synchronous binary counter tracking threshold-crossing pulses.
//              Asserts anomaly_limit_reached when count exceeds limit.
// =============================================================================

`timescale 1ns / 1ps

module abnormal_counter #(
    parameter COUNT_WIDTH   = 16,
    parameter ANOMALY_LIMIT = 5
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   count_enable,   // Connected to threshold_exceeded
    input  wire                   clear_count,    // Synchronous clear signal
    
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
                // Increment on threshold violation
                abnormal_count <= abnormal_count + 1'b1;
            end
        end
    end

    // Flag when abnormal events exceed our tolerance threshold
    assign anomaly_limit_reached = (abnormal_count >= ANOMALY_LIMIT);

endmodule
