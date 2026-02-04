#!/usr/bin/env python3
"""
Simple script to analyze chat parameter sweep results
"""

import json
import pandas as pd
from analyze_results import analyze_text_quality

def analyze_chat_results():
    """Analyze chat results with text quality metrics"""
    
    try:
        print('🔍 ANALYZING CHAT PARAMETER SWEEP RESULTS')
        print('=' * 50)
        
        # Load optimized chat results
        with open('chat_parameter_sweep_optimized_20260128_181510.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f'📊 Loaded {len(results)} chat responses')
        
        # Analyze each response
        analyzed_results = []
        
        for i, result in enumerate(results):
            if i % 10 == 0:
                print(f'📋 Processing response {i+1}/{len(results)}...')
            
            try:
                response_text = result.get('response', '')
                if isinstance(response_text, str):
                    # Calculate quality metrics
                    quality_metrics = analyze_text_quality(response_text)
                    
                    # Create analyzed result
                    analyzed_result = {
                        'prompt': result.get('prompt', ''),
                        'parameters': result.get('parameters', {}),
                        'response': response_text,
                        'response_length': len(response_text),
                        'success': result.get('success', False),
                        **quality_metrics
                    }
                    analyzed_results.append(analyzed_result)
            except Exception as e:
                print(f'⚠️  Error analyzing response {i+1}: {e}')
        
        # Create DataFrame
        df = pd.DataFrame(analyzed_results)
        
        print(f'\n✅ Analysis complete!')
        print(f'📈 Generated DataFrame with {len(df)} rows and {len(df.columns)} columns')
        
        # Summary statistics
        print(f'\n📊 SUMMARY STATISTICS:')
        print(f'   Average Quality Score: {df["quality_score"].mean():.3f}')
        print(f'   Average Repetition Penalty: {df["repetition_penalty"].mean():.3f}')
        print(f'   Average Coherence Score: {df["coherence_score"].mean():.3f}')
        print(f'   Average Readability Score: {df["readability_score"].mean():.3f}')
        print(f'   Average Response Length: {df["response_length"].mean():.1f} characters')
        
        # Calculate overall score
        df['overall_score'] = (
            df['quality_score'] * 0.4 +
            (1 - df['repetition_penalty']) * 0.3 +
            df['coherence_score'] * 0.2 +
            df['readability_score'] * 0.1
        )
        
        # Find top parameter combinations (use actual column names from parameters)
        # First, extract parameters to separate columns
        if 'parameters' in df.columns and isinstance(df['parameters'].iloc[0], dict):
            params_df = df['parameters'].apply(pd.Series)
            df = pd.concat([df, params_df], axis=1)
        
        grouped = df.groupby(['temperature', 'min_p', 'adaptive_target']).agg({
            'quality_score': 'mean',
            'repetition_penalty': 'mean',
            'coherence_score': 'mean',
            'readability_score': 'mean',
            'overall_score': 'mean',
            'response_length': 'mean'
        }).reset_index()
        
        top_combinations = grouped.nlargest(5, 'overall_score')
        
        print(f'\n🎯 TOP 5 PARAMETER COMBINATIONS:')
        for i, row in top_combinations.iterrows():
            print(f'{i+1}. temp={row["temperature"]:.2f}, min_p={row["min_p"]:.3f}, adapt={row["adaptive_target"]:.2f}')
            print(f'   Overall: {row["overall_score"]:.3f}, Quality: {row["quality_score"]:.3f}')
            print(f'   Repetition: {row["repetition_penalty"]:.3f}, Coherence: {row["coherence_score"]:.3f}')
            print(f'   Avg Length: {row["response_length"]:.1f} chars')
        
        # Save results
        output_filename = 'chat_parameter_sweep_optimized_analyzed.json'
        df.to_json(output_filename, orient='records', indent=2)
        
        csv_filename = 'chat_parameter_sweep_optimized_analyzed.csv'
        df.to_csv(csv_filename, index=False)
        
        print(f'\n💾 Results saved:')
        print(f'   JSON: {output_filename}')
        print(f'   CSV: {csv_filename}')
        
        # Show some sample analyzed responses
        print(f'\n📝 SAMPLE ANALYZED RESPONSES:')
        for i, sample in enumerate(df.nlargest(3, 'overall_score').iterrows()):
            idx, row = sample
            print(f'\nSample {i+1} (Overall Score: {row["overall_score"]:.3f}):')
            print(f'Parameters: temp={row["temperature"]:.2f}, min_p={row["min_p"]:.3f}, adapt={row["adaptive_target"]:.2f}')
            print(f'Response: {results[idx]["response"][:150]}...')
        
        return True
        
    except Exception as e:
        print(f'❌ Error analyzing chat results: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_chat_results()