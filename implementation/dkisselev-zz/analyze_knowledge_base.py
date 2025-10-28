"""Analyze knowledge base for optimal chunking strategy"""
import os
import glob
from pathlib import Path
import statistics

KNOWLEDGE_BASE = "knowledge-base"

# Analyze all markdown files
files_by_type = {}
for folder in glob.glob(f"{KNOWLEDGE_BASE}/*"):
    doc_type = os.path.basename(folder)
    files = glob.glob(f"{folder}/**/*.md", recursive=True)
    
    file_stats = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            file_stats.append({
                'path': file_path,
                'chars': len(content),
                'lines': len(lines),
                'words': len(content.split())
            })
    
    files_by_type[doc_type] = file_stats

# Print analysis
total_files = 0
total_chars = 0

for doc_type, files in sorted(files_by_type.items()):
    print(f"\n{doc_type.upper()}: {len(files)} files")
    print("-" * 80)
    
    chars = [f['chars'] for f in files]
    words = [f['words'] for f in files]
    lines = [f['lines'] for f in files]
    
    total_files += len(files)
    total_chars += sum(chars)
    
    print(f"  Chars:  min={min(chars):,}  avg={statistics.mean(chars):,.0f}  max={max(chars):,}")
    print(f"  Words:  min={min(words):,}  avg={statistics.mean(words):,.0f}  max={max(words):,}")
    print(f"  Lines:  min={min(lines):,}  avg={statistics.mean(lines):,.0f}  max={max(lines):,}")
    
    # Size distribution
    size_buckets = {'<1k': 0, '1-2k': 0, '2-3k': 0, '3-5k': 0, '>5k': 0}
    for c in chars:
        if c < 1000:
            size_buckets['<1k'] += 1
        elif c < 2000:
            size_buckets['1-2k'] += 1
        elif c < 3000:
            size_buckets['2-3k'] += 1
        elif c < 5000:
            size_buckets['3-5k'] += 1
        else:
            size_buckets['>5k'] += 1
    
    print(f"  Size Distribution: {size_buckets}")
    
    # Sample a few files
    if files:
        print(f"\n  Sample files:")
        for f in files[:3]:
            name = Path(f['path']).name
            print(f"    {name}: {f['chars']:,} chars, {f['words']:,} words")

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print(f"Total files: {total_files}")
print(f"Total chars: {total_chars:,}")
print(f"Average file size: {total_chars/total_files:,.0f} chars")

print("\n" + "=" * 80)
print("CHUNKING RECOMMENDATIONS")
print("=" * 80)
print(f"\nAnalysis:")
all_chars = [f['chars'] for files in files_by_type.values() for f in files]
avg_size = statistics.mean(all_chars)
median_size = statistics.median(all_chars)

print(f"  Avg file size: {avg_size:.0f} chars")
print(f"  Median file size: {median_size:.0f} chars")

if avg_size < 2000:
    print(f"\nFiles are small (avg {avg_size:.0f} chars)")
    print(f"  Recommendation: chunk_size=800-1000t")
else:
    print(f"\nFiles are medium-large (avg {avg_size:.0f} chars)")
    print(f"  Recommendation: chunk_size=1200-1500")


