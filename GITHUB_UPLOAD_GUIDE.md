# GitHub Upload Guide

Step-by-step instructions for uploading this project to GitHub.

## ✅ Pre-Upload Checklist

All cleanup is complete:
- [x] No accidental files (nul deleted)
- [x] No __pycache__ directories
- [x] No buggy scripts (reinstall_dependencies removed)
- [x] Proper attribution in README.md
- [x] LICENSE file created
- [x] CHANGELOG.md created
- [x] .gitignore configured (.claude/, CLAUDE.md, *.pkl)
- [x] Documentation complete

## 📋 Upload Strategy: Don't Track Large Model

**Decision:** The 295MB model file will NOT be uploaded to GitHub.
- ✅ Small AMPS model (1.2 KB) will be included
- ❌ Large LUPE model (295 MB) will be excluded
- Users download from Box link (already in README)

---

## 🚀 Step-by-Step Upload Instructions

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in repository details:
   - **Repository name**: `LUPE-analysis-tool` (or your preference)
   - **Description**: "A GUI and CLI tool for LUPE behavioral analysis - converts Jupyter notebooks to standalone application"
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we have these already)
3. Click **"Create repository"**
4. Copy the repository URL (e.g., `https://github.com/YOUR_USERNAME/LUPE-analysis-tool.git`)

---

### Step 2: Initialize Git Repository

Open terminal in project directory and run:

```bash
cd "C:\Users\rasandor\Documents\Repositories\LUPE_analysis_RASO_version"

# Initialize git repository
git init

# Set default branch to main
git branch -M main
```

---

### Step 3: Add Files to Git

```bash
# Add all files (respects .gitignore)
git add .

# Force-add only the small AMPS model (1.2 KB)
git add -f models/model_AMPS.pkl

# Verify what will be committed
git status
```

**Expected output:**
- ✅ CLAUDE.md should NOT appear (excluded by .gitignore)
- ✅ model_AMPS.pkl should appear (force-added)
- ❌ model_LUPE.pkl should NOT appear (excluded by .gitignore)
- ✅ All other project files should appear

If you see `models/model_LUPE.pkl` in the list, STOP and run:
```bash
git reset models/model_LUPE.pkl
```

---

### Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: LUPE Analysis Tool v1.0.0

- Main launcher GUI with LUPE and AMPS options
- LUPE classification GUI for behavior analysis
- LUPE-AMPS pain scale analysis GUI
- CLI interface for automation
- Comprehensive documentation and guides
- JSON-based configuration system
- No Jupyter notebooks required

Based on LUPE 2.0 by Corder Lab (UPenn) & Yttri Lab (CMU)"
```

---

### Step 5: Connect to GitHub and Push

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub details:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

**If prompted for credentials:**
- Username: Your GitHub username
- Password: Use a Personal Access Token (not your password)
  - Create token at: https://github.com/settings/tokens
  - Required scopes: `repo` (full control)

---

### Step 6: Verify Upload

1. Go to your GitHub repository in browser
2. Check that files are present:
   - ✅ README.md displays correctly
   - ✅ LICENSE file is visible
   - ✅ models/model_AMPS.pkl is present (1.2 KB)
   - ❌ models/model_LUPE.pkl is NOT present (good!)
   - ❌ CLAUDE.md is NOT present (good!)
3. Check repository size (should be ~5-10 MB, not 300+ MB)

---

## 📝 Post-Upload Notes

### What Users Will See

When users clone your repository:
```bash
git clone https://github.com/YOUR_USERNAME/LUPE-analysis-tool.git
cd LUPE-analysis-tool
pip install -r requirements.txt

# They'll need to download the large model from Box
# (README.md already provides the link)
```

### Model Download Instructions (Already in README)

Your README.md already includes download instructions at line 102:
```markdown
5. **Download the LUPE A-SOiD model**
   - Download from: [LUPE Model Link](https://upenn.box.com/s/9rfslrvcc7m6fji8bmgktnegghyu88b0)
   - Place the model file in a `models/` directory
```

---

## 🔧 Troubleshooting

### "Large files detected" Error

If you accidentally try to push model_LUPE.pkl:
```bash
# Remove from git tracking
git rm --cached models/model_LUPE.pkl

# Commit the removal
git commit -m "Remove large model from tracking"

# Push again
git push
```

### Authentication Failed

Use a Personal Access Token instead of password:
1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Copy token and use as password

Or use GitHub CLI:
```bash
# Install GitHub CLI
winget install GitHub.cli

# Authenticate
gh auth login

# Push
git push
```

### .gitignore Not Working

If files that should be ignored are tracked:
```bash
# Clear git cache
git rm -r --cached .

# Re-add files (respects .gitignore)
git add .

# Commit
git commit -m "Fix .gitignore"
```

---

## ✨ Next Steps After Upload

1. **Add repository description and topics**
   - Go to repository settings
   - Add topics: `behavior-analysis`, `neuroscience`, `python`, `gui`, `deeplabcut`

2. **Enable GitHub Pages** (optional)
   - Host documentation at `username.github.io/LUPE-analysis-tool`

3. **Create releases** (optional)
   - Tag version 1.0.0
   - Attach pre-built executables (future)

4. **Set up GitHub Actions** (optional)
   - Automated testing
   - Code quality checks

---

## 📞 Questions?

See the main README.md for project documentation and troubleshooting.

**Ready to upload!** 🚀
