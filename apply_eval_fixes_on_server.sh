#!/bin/bash
# Apply evaluation fixes directly on the server
# Run this script on the server (willi) to fix the evaluation scripts

set -e

echo "=== Applying Evaluation Fixes ==="
echo ""

# Navigate to project directory
cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40/hybrid_model_mamba_xlstm

echo "1. Backing up original files..."
cp scripts/evaluate_sts.py scripts/evaluate_sts.py.backup
cp scripts/evaluate_retrieval.py scripts/evaluate_retrieval.py.backup
echo "   ✓ Backups created"

echo ""
echo "2. Applying fixes to evaluate_sts.py..."

# Fix evaluate_sts.py - Replace the broken load_encoder_from_checkpoint function
python3 << 'EOF'
import re

with open('scripts/evaluate_sts.py', 'r') as f:
    content = f.read()

# Fix 1: Change default dim from 768 to 512
content = re.sub(
    r'dim = 768  # default for 70M model',
    'dim = 512  # 70M model uses 512, not 768',
    content
)

# Fix 2: Change default num_layers from 12 to 8
content = re.sub(
    r'num_layers = 12  # default',
    'num_layers = 8  # 70M model has 8 layers, not 12',
    content
)

# Fix 3: Move layer counting before prefix stripping
old_pattern = r'''    # Extract state dict and strip prefixes
    raw_state_dict = ckpt\.get\("state_dict", ckpt\)
    state_dict = \{\}
    
    for k, v in raw_state_dict\.items\(\):
        # Remove Lightning module prefix first
        if k\.startswith\("model\."\):
            k = k\[len\("model\."\):\]
        
        # Remove lm\. prefix for encoder \(training uses HybridLanguageModel with lm\. prefix\)
        if k\.startswith\("lm\."\):
            k = k\[len\("lm\."\):\]
        
        # Skip projection head and logit_scale \(not needed for encoder\)
        if k\.startswith\("projection_head\."\) or k == "logit_scale":
            continue
            
        state_dict\[k\] = v
    
    # Infer config from checkpoint
    dim = 512  # 70M model uses 512, not 768
    for k, v in state_dict\.items\(\):
        if "token_embedding\.weight" in k:
            dim = int\(v\.shape\[1\]\)
            break
    
    # Count layers
    num_layers = 0
    for k in state_dict\.keys\(\):
        if "lm\.layers\." in k:
            import re
            m = re\.search\(r"lm\\\.layers\\\.\\(\\d\+\\)\\.", k\)
            if m:
                idx = int\(m\.group\(1\)\)
                if idx \+ 1 > num_layers:
                    num_layers = idx \+ 1
    
    if num_layers == 0:
        num_layers = 8  # 70M model has 8 layers, not 12'''

new_pattern = '''    # Extract state dict
    raw_state_dict = ckpt.get("state_dict", ckpt)
    
    # Count layers BEFORE stripping prefixes
    import re
    num_layers = 0
    for k in raw_state_dict.keys():
        m = re.search(r"layers\\.(\d+)\\.", k)
        if m:
            idx = int(m.group(1))
            if idx + 1 > num_layers:
                num_layers = idx + 1
    
    if num_layers == 0:
        num_layers = 8  # 70M model has 8 layers, not 12
    
    # Now strip prefixes
    state_dict = {}
    for k, v in raw_state_dict.items():
        # Remove Lightning module prefix first
        if k.startswith("model."):
            k = k[len("model."):]
        
        # Remove lm. prefix for encoder (training uses HybridLanguageModel with lm. prefix)
        if k.startswith("lm."):
            k = k[len("lm."):]
        
        # Skip projection head and logit_scale (not needed for encoder)
        if k.startswith("projection_head.") or k == "logit_scale":
            continue
            
        state_dict[k] = v
    
    # Infer config from checkpoint
    dim = 512  # 70M model uses 512, not 768
    for k, v in state_dict.items():
        if "token_embedding.weight" in k:
            dim = int(v.shape[1])
            break'''

content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)

with open('scripts/evaluate_sts.py', 'w') as f:
    f.write(content)

print("   ✓ evaluate_sts.py fixed")
EOF

echo ""
echo "3. Applying fixes to evaluate_retrieval.py..."

# Fix evaluate_retrieval.py - Same fixes plus retrieval pairs fix
python3 << 'EOF'
import re

with open('scripts/evaluate_retrieval.py', 'r') as f:
    content = f.read()

# Fix 1: Change default dim from 768 to 512
content = re.sub(
    r'dim = 768  # default for 70M model',
    'dim = 512  # 70M model uses 512, not 768',
    content
)

# Fix 2: Change default num_layers from 12 to 8
content = re.sub(
    r'num_layers = 12  # default',
    'num_layers = 8  # 70M model has 8 layers, not 12',
    content
)

# Fix 3: Fix retrieval pairs - replace consecutive abstracts with title→abstract
old_retrieval = r'''        abstracts = \[\]
        for i, item in enumerate\(dataset\):
            if i >= num_pairs \* 2:  # Need 2x for pairs
                break
            
            # This dataset has 'article' and 'abstract' fields
            article = item\.get\("article", ""\)
            
            if article and len\(article\) > 50:  # Ensure meaningful text
                abstracts\.append\(article\)
        
        # Create pairs \(consecutive abstracts\)
        pairs = \[\]
        for i in range\(0, len\(abstracts\) - 1, 2\):
            if len\(pairs\) >= num_pairs:
                break
            pairs\.append\(\(abstracts\[i\], abstracts\[i \+ 1\]\)\)'''

new_retrieval = '''        pairs = []
        for i, item in enumerate(dataset):
            if len(pairs) >= num_pairs:
                break
            
            # This dataset has 'article' and 'abstract' fields
            article = item.get("article", "")
            abstract = item.get("abstract", "")
            
            if article and abstract and len(article) > 50 and len(abstract) > 50:
                # Correct: pair title with its own abstract
                # Truncate article to simulate title/beginning
                pairs.append((article[:200], abstract))'''

content = re.sub(old_retrieval, new_retrieval, content, flags=re.DOTALL)

with open('scripts/evaluate_retrieval.py', 'w') as f:
    f.write(content)

print("   ✓ evaluate_retrieval.py fixed")
EOF

echo ""
echo "4. Clearing Python cache..."
find scripts -name "*.pyc" -delete 2>/dev/null || true
find scripts -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ Cache cleared"

echo ""
echo "=== Fixes Applied Successfully ==="
echo ""
echo "Backups saved as:"
echo "  - scripts/evaluate_sts.py.backup"
echo "  - scripts/evaluate_retrieval.py.backup"
echo ""
echo "Next steps:"
echo "  1. Verify fixes: python verify_eval_fixes.py --checkpoint <your_checkpoint>"
echo "  2. Re-run evaluation: sbatch scripts/eval_stage1_gpu0.sh"
