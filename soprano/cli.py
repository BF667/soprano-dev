#!/usr/bin/env python3
"""
SopranoTTS CLI - Command-line interface for Soprano text-to-speech
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(
        description='SopranoTTS CLI - Generate speech from text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - save to file
  soprano-tts "Hello world" -o hello.wav
  
  # Batch processing
  soprano-tts "First sentence." "Second sentence." --batch-output ./output_dir
  
  # Streaming to stdout
  soprano-tts "Streaming audio" --stream --stream-chunk-size 2 | aplay
  
  # Advanced parameters
  soprano-tts "Text here" -o output.wav --temperature 0.5 --top-p 0.9 --repetition-penalty 1.1
  
  # Use CPU and transformers backend
  soprano-tts "Text" -o cpu.wav --device cpu --backend transformers
  
  # Verbose output
  soprano-tts "Hello" -o test.wav -v
        """
    )
    
    # Text input arguments
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        'text',
        nargs='*',
        help='Text to synthesize (multiple texts for batch mode)'
    )
    text_group.add_argument(
        '--text-file', '-f',
        type=str,
        help='File containing text to synthesize (one per line)'
    )
    text_group.add_argument(
        '--stdin',
        action='store_true',
        help='Read text from stdin'
    )
    
    # Output arguments
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (for single text)'
    )
    output_group.add_argument(
        '--batch-output', '-b',
        type=str,
        help='Output directory for batch processing'
    )
    output_group.add_argument(
        '--stream',
        action='store_true',
        help='Stream audio to stdout as raw PCM'
    )
    
    # Model parameters
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['auto', 'lmdeploy', 'transformers'],
        default='auto',
        help='Backend to use for inference'
    )
    parser.add_argument(
        '--cache-size-mb',
        type=int,
        default=10,
        help='Cache size in MB (for lmdeploy backend)'
    )
    parser.add_argument(
        '--decoder-batch-size',
        type=int,
        default=1,
        help='Batch size for decoder inference'
    )
    
    # Generation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.2,
        help='Repetition penalty'
    )
    parser.add_argument(
        '--min-sentence-length',
        type=int,
        default=30,
        help='Minimum sentence length before merging'
    )
    
    # Streaming parameters
    parser.add_argument(
        '--stream-chunk-size',
        type=int,
        default=1,
        help='Chunk size for streaming (number of tokens per chunk)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available devices and exit'
    )
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("Available devices:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  cuda:{i} - {torch.cuda.get_device_name(i)}")
        print("  cpu")
        sys.exit(0)
    
    # Get text input
    texts = []
    
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading text file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.stdin:
        texts = [sys.stdin.read().strip()]
    else:
        texts = args.text
    
    if not texts:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)
    
    # Validate output options
    if len(texts) > 1 and args.output:
        print("Error: --output cannot be used with multiple texts. Use --batch-output instead.", file=sys.stderr)
        sys.exit(1)
    
    if args.batch_output and len(texts) == 1:
        print("Warning: --batch-output specified but only one text provided.", file=sys.stderr)
    
    if args.stream and (args.output or args.batch_output):
        print("Error: --stream cannot be used with --output or --batch-output", file=sys.stderr)
        sys.exit(1)
    
    # Run inference
    try:
        if args.verbose:
            print(f"Initializing SopranoTTS...")
            print(f"  Device: {args.device}")
            print(f"  Backend: {args.backend}")
            print(f"  Texts: {len(texts)}")
        
        # Import here to avoid loading model when just listing devices
        from soprano.tts import SopranoTTS
        
        tts = SopranoTTS(
            backend=args.backend,
            device=args.device,
            cache_size_mb=args.cache_size_mb,
            decoder_batch_size=args.decoder_batch_size
        )
        
        # Single text with file output
        if args.output:
            if args.verbose:
                print(f"Generating single audio to: {args.output}")
            
            tts.infer(
                text=texts[0],
                out_path=args.output,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
            
            if args.verbose:
                print(f"Audio saved to: {args.output}")
        
        # Batch processing
        elif args.batch_output:
            if args.verbose:
                print(f"Generating batch audio to: {args.batch_output}")
            
            tts.infer_batch(
                texts=texts,
                out_dir=args.batch_output,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
            
            if args.verbose:
                print(f"Generated {len(texts)} audio files in: {args.batch_output}")
        
        # Streaming
        elif args.stream:
            if len(texts) > 1:
                print("Warning: Only first text will be used for streaming", file=sys.stderr)
            
            if args.verbose:
                print(f"Streaming audio (chunk size: {args.stream_chunk_size})")
                print("Writing raw 32kHz 16-bit PCM to stdout...")
            
            # Write raw PCM to stdout
            import numpy as np
            
            for chunk in tts.infer_stream(
                text=texts[0],
                chunk_size=args.stream_chunk_size,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            ):
                # Convert to int16 and write to stdout
                audio_int16 = (chunk.numpy() * 32767).astype(np.int16)
                sys.stdout.buffer.write(audio_int16.tobytes())
        
        # No output specified - generate single and save to default
        else:
            if len(texts) > 1:
                print("Warning: Multiple texts provided but no output specified. Using first text only.", file=sys.stderr)
            
            default_output = "output.wav"
            if args.verbose:
                print(f"Generating audio to default file: {default_output}")
            
            tts.infer(
                text=texts[0],
                out_path=default_output,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
            
            if args.verbose:
                print(f"Audio saved to: {default_output}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
