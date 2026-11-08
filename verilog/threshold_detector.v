// =============================================================================
// Module Name: threshold_detector
// Project:     AI-Driven RF Signal Classification & Interference Detection System
// Phase:       Step 4 - Digital System Design (Module 3 of 6)
// Description: High-speed 32-bit digital magnitude comparator.
//              Compares signal power P against a programmable threshold P_th.
// =============================================================================

`timescale 1ns / 1ps

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
