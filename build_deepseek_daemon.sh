#!/bin/bash
# Build and install script for Swift ASI-DEEPSEEK Daemon (Upgraded from QWEN)

set -e

# Configuration
PROJECT_DIR="/Users/carolalvarez/Applications/Allentown-L104-Node"
SOURCE_FILE="$PROJECT_DIR/ASIDeepSeekDaemon.swift"
BUILD_DIR="$PROJECT_DIR/build"
OUTPUT_BINARY="$BUILD_DIR/asi-deepseek-daemon"
LOG_DIR="$PROJECT_DIR/logs"

echo "🚀 Building Swift ASI-DEEPSEEK Daemon..."

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Compile the Swift daemon
echo "🔨 Compiling Swift code..."
swiftc -O -o "$OUTPUT_BINARY" "$SOURCE_FILE"

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "📦 Binary created at: $OUTPUT_BINARY"
    
    # Set executable permissions
    chmod +x "$OUTPUT_BINARY"
    
    # Create launchd plist file
    cat > "$BUILD_DIR/com.l104.asi-deepseek-swift.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- L104 Swift ASI-DEEPSEEK Background Upgrade Worker -->
    <key>Label</key>
    <string>com.l104.asi-deepseek-swift</string>

    <!-- Program to execute -->
    <key>ProgramArguments</key>
    <array>
        <string>$OUTPUT_BINARY</string>
    </array>

    <!-- Working directory -->
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <!-- Run as user -->
    <key>UserName</key>
    <string>carolalvarez</string>

    <!-- Keep running -->
    <key>KeepAlive</key>
    <true/>

    <!-- Run on login -->
    <key>RunAtLoad</key>
    <true/>

    <!-- Standard output/error logging -->
    <key>StandardOutPath</key>
    <string>$LOG_DIR/asi_deepseek_swift.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/asi_deepseek_swift_error.log</string>

    <!-- Environment variables -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>L104_NODE_PATH</key>
        <string>$PROJECT_DIR</string>
    </dict>

    <!-- Nice level (lower priority) -->
    <key>Nice</key>
    <integer>10</integer>

    <!-- Soft resource limits -->
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>1024</integer>
        <key>ResidentSetSizeMB</key>
        <integer>2048</integer>
    </dict>

    <!-- Start interval (seconds) - throttle restarts -->
    <key>ThrottleInterval</key>
    <integer>60</integer>
</dict>
</plist>
EOF

    echo "📋 Launchd plist created at: $BUILD_DIR/com.l104.asi-deepseek-swift.plist"
    
    # Install the launchd service
    echo "🔧 Installing launchd service..."
    mkdir -p ~/Library/LaunchAgents
    cp "$BUILD_DIR/com.l104.asi-deepseek-swift.plist" ~/Library/LaunchAgents/
    
    # Unload existing service if running
    launchctl unload ~/Library/LaunchAgents/com.l104.asi-deepseek-swift.plist 2>/dev/null || true
    
    # Load the new service
    launchctl load ~/Library/LaunchAgents/com.l104.asi-deepseek-swift.plist
    
    echo "✅ Swift ASI-DEEPSEEK Daemon installed and started!"
    echo "📊 To check status: launchctl list | grep l104.asi-deepseek-swift"
    echo "📜 Logs available at: $LOG_DIR/asi_deepseek_swift.log"
    
else
    echo "❌ Compilation failed!"
    exit 1
fi
