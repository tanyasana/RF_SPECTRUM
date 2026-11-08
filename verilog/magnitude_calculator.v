// =============================================================================
// Module Name: magnitude_calculator
// Project:     AI-Driven RF Signal Classification & Interference Detection System
// Phase:       Step 4 - Digital System Design (Module 2 of 6)
// Description: Computes Instantaneous Signal Power P = I^2 + Q^2
//              Uses 16-bit signed multipliers and 32-bit addition.
//              Pipelined with 1-cycle latency for high clock frequency.
// =============================================================================

`timescale 1ns / 1ps

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

    // Internal signed multiplication terms (16-bit x 16-bit = 32-bit)
    wire signed [2*DATA_WIDTH-1:0] i_mult = in_I * in_I;
    wire signed [2*DATA_WIDTH-1:0] q_mult = in_Q * in_Q;

    // Sequential pipeline register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            power_out <= {POWER_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end
        else begin
            valid_out <= valid_in;
            if (valid_in) begin
                // Sum of squares: P = I^2 + Q^2
                power_out <= i_mult[POWER_WIDTH-1:0] + q_mult[POWER_WIDTH-1:0];
            end
            else begin
                power_out <= {POWER_WIDTH{1'b0}};
            end
        end
    end

endmodule
