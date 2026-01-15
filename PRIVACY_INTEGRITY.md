# L104 Sovereign Privacy & Integrity Protocol

To maintain the **God-Code** integrity and maximize privacy, the following configurations must be enforced across all Sovereign Nodes.

## 1. Local Git Configuration (Privacy & Integrity)

Run these commands in your terminal to align your local environment with the Sovereign standard:

```bash
# Force linear history (Rebase on pull)
git config --global pull.rebase true

# Enforce commit integrity (GPG Signing)
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# Privacy: Use GitHub private email
# Replace with your actual private email (found in GitHub Settings -> Emails)
git config --global user.email "id+username@users.noreply.github.com"

# Maintenance: Prune stale branches automatically
git config --global fetch.prune true

# Push tags automatically with commits
git config --global push.followTags true
```

## 2. GPG Signing (The God-Code Signature)

Every commit must be signed to verify it originates from an authorized Sovereign entity.

### Step 1: Generate your GPG key
```bash
gpg --full-generate-key
# Select: (1) RSA and RSA (default)
# Keysize: 4096 bits
# Validity: Does not expire (or as required)
```

### Step 2: Export your Public Key
Find your Key ID:
```bash
gpg --list-secret-keys --keyid-format=LONG
```
Copy the ID (e.g., `3AA5C34371567BD2`) and export it:
```bash
gpg --armor --export 3AA5C34371567BD2
```

### Step 3: Add to GitHub
1. Copy the output from Step 2.
2. Go to **Settings -> SSH and GPG keys -> New GPG key**.
3. Paste the key.

### Step 4: Configure Git to use your key
```bash
git config --global user.signingkey 3AA5C34371567BD2
```

## 3. GitHub Repository Protection Rules

To lock the `main` branch, go to **Settings -> Branches -> Add branch protection rule**:

- **Branch name pattern**: `main`
- [x] **Require a pull request before merging**
  - [x] **Require approvals** (Set to 1, limited to `@lockephi` via CODEOWNERS)
- [x] **Require signed commits** (Enforces GPG verification)
- [x] **Require linear history** (Prevents merge commits, enforces rebase)
- [x] **Lock branch** (Branch is read-only unless specifically unlocked)
- [x] **Do not allow bypassing the above settings** (Even admins are restricted)

## 4. Privacy Settings
- Go to **Settings -> Emails** and check **Keep my email addresses private**.
- Go to **Settings -> Emails** and check **Block command line pushes that expose my email**.
