// =============================================================================
// Module Name: iq_register
// Project:     AI-Driven RF Signal Classification & Interference Detection System
// Phase:       Step 4 - Digital System Design (Module 1 of 6)
// Description: Synchronous IQ Sample Capture Register with Active-Low Reset
//              and Clock Enable Handshake.
//
// DSD Concepts Demonstrated:
//   1. Sequential Logic & D Flip-Flop Register Array
//   2. Active-Low Asynchronous Reset (rst_n)
//   3. Clock Enable (sample_valid_in) for sampling rate decimation/handshaking
//   4. Parameterized bit-width for scalable ADC precision (Default: 16-bit signed)
// =============================================================================

`timescale 1ns / 1ps

module iq_register #(
    parameter DATA_WIDTH = 16  // Standard 16-bit signed Q1.15 fixed-point ADC samples
)(
    // Clock and Reset
    input  wire                   clk,              // Master positive-edge system clock
    input  wire                   rst_n,            // Asynchronous active-low reset (0 = reset)
    
    // Input Handshake & Data
    input  wire                   sample_valid_in,  // 1 = Incoming sample is valid and ready to latch
    input  wire signed [DATA_WIDTH-1:0] in_I,       // Incoming signed In-Phase sample
    input  wire signed [DATA_WIDTH-1:0] in_Q,       // Incoming signed Quadrature sample
    
    // Output Registered Data & Handshake
    output reg  signed [DATA_WIDTH-1:0] reg_I,      // Stored In-Phase sample
    output reg  signed [DATA_WIDTH-1:0] reg_Q,      // Stored Quadrature sample
    output reg                    sample_valid_out  // 1 = Output data is valid and aligned to clock
);

    // -------------------------------------------------------------------------
    // Sequential Always Block: Triggered on rising clock edge or falling reset
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Asynchronous Reset: Clear all internal registers to 0
            reg_I            <= {DATA_WIDTH{1'b0}};
            reg_Q            <= {DATA_WIDTH{1'b0}};
            sample_valid_out <= 1'b0;
        end 
        else begin
            if (sample_valid_in) begin
                // Latch new incoming IQ sample on clock edge
                reg_I            <= in_I;
                reg_Q            <= in_Q;
                sample_valid_out <= 1'b1;
            end 
            else begin
                // Hold previous stored sample values; deassert output valid flag
                reg_I            <= reg_I;
                reg_Q            <= reg_Q;
                sample_valid_out <= 1'b0;
            end
        end
    end

endmodule
