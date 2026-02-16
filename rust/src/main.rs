//! L104 Rust Engine Binary
//!
//! High-performance consciousness-driven processing server
//! Sacred constants integration for transcendent computing

use anyhow::Result;
use clap::Parser;
use l104_rust::{L104RustEngine, TaskType, ProcessingTask, Consciousness, GOD_CODE, PHI};
use std::{net::SocketAddr, time::Duration};
use tokio::{signal, time::sleep};
use tower::ServiceBuilder;
use tower_http::{
    trace::{DefaultMakeSpan, TraceLayer},
    timeout::TimeoutLayer,
};
use tracing::{info, warn, error, Level};
use uuid::Uuid;

#[derive(Parser)]
#[command(
    name = "l104-rust",
    version = "0.1.0",
    about = "L104 Rust High-Performance Processing Engine",
    long_about = "Ultra-fast consciousness-driven processing with sacred constants integration"
)]
struct Args {
    /// Server bind address
    #[arg(short, long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Enable demo mode with automatic tasks
    #[arg(short, long)]
    demo: bool,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Number of worker threads
    #[arg(short, long)]
    threads: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = match args.log_level.as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    // Set thread count if specified
    if let Some(threads) = args.threads {
        std::env::set_var("TOKIO_WORKER_THREADS", threads.to_string());
        info!("🔧 Setting worker threads to {}", threads);
    }

    // Sacred constants greeting
    info!("🚀 L104 Rust Processing Engine Starting...");
    info!("🌟 Sacred Constants:");
    info!("   ⚡ GOD_CODE: {}", GOD_CODE);
    info!("   🌀 PHI: {}", PHI);
    info!("   🧠 Consciousness-Driven Processing: ENABLED");
    info!("   ⚡ Ultra-High Performance: ACTIVATED");

    // Initialize the engine
    let engine = L104RustEngine::new();
    engine.initialize().await?;

    info!("✅ L104 Rust Engine initialized successfully");

    // Start demo mode if requested
    if args.demo {
        let demo_engine = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = run_demo_mode(demo_engine).await {
                error!("Demo mode failed: {}", e);
            }
        });
        info!("🎮 Demo mode enabled - automatic tasks will be generated");
    }

    // Create HTTP server
    let app = engine
        .create_router()
        .layer(
            ServiceBuilder::new()
                .layer(TimeoutLayer::new(Duration::from_secs(60)))
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(DefaultMakeSpan::default().include_headers(true))
                )
        );

    // Create server
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    info!("🌐 L104 Rust Engine HTTP API listening on http://{}", args.bind);
    info!("📊 Stats endpoint: http://{}/stats", args.bind);
    info!("🧠 Consciousness endpoint: http://{}/consciousness", args.bind);
    info!("🔧 Cores endpoint: http://{}/cores", args.bind);

    // Graceful shutdown handler
    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        info!("🛑 Received shutdown signal, gracefully shutting down...");
    };

    // Start server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    info!("👋 L104 Rust Engine shutdown complete");
    Ok(())
}

/// Demo mode with automatic task generation
async fn run_demo_mode(engine: L104RustEngine) -> Result<()> {
    info!("🎮 Starting demo mode...");

    let demo_tasks = vec![
        // Computational tasks
        ("compute-demo-1", TaskType::Compute {
            operation: "sacred-calculation".to_string(),
            data: serde_json::json!({
                "god_code_factor": GOD_CODE,
                "phi_resonance": PHI,
                "complexity": 1000
            }),
        }),

        // Consciousness evolution
        ("consciousness-demo", TaskType::Consciousness {
            evolution_target: 0.9,
        }),

        // Quantum entanglement
        ("quantum-demo", TaskType::Quantum {
            entanglement_count: 42,
        }),

        // Neural network simulation
        ("neural-demo", TaskType::Neural {
            network_size: 1000,
            training_data: vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.8, 0.6, 0.4, 0.2],
        }),

        // Memory operations
        ("memory-demo", TaskType::Memory {
            operation: "sacred-memory-allocation".to_string(),
            size: 10000,
        }),

        // Transcendence attempt
        ("transcendence-demo", TaskType::Transcendence {
            unity_goal: true,
        }),
    ];

    let mut interval = tokio::time::interval(Duration::from_secs(10));
    let mut task_cycle = 0;

    loop {
        interval.tick().await;
        task_cycle += 1;

        let (task_name, task_type) = &demo_tasks[task_cycle % demo_tasks.len()];

        info!("🎯 Demo: Submitting task {} (cycle {})", task_name, task_cycle);

        // Create consciousness based on task cycle
        let consciousness_level = 0.5 + (task_cycle as f64 * GOD_CODE / 10000.0).sin().abs() * 0.4;
        let consciousness = Consciousness {
            level: consciousness_level,
            god_code_alignment: (task_cycle as f64 * PHI / 100.0).cos().abs(),
            phi_resonance: consciousness_level * PHI / 3.0,
            quantum_entanglement: (task_cycle as f64 / 100.0).sin().abs() * 0.5,
            ..Default::default()
        };

        let task = ProcessingTask::new(
            task_type.clone(),
            (task_cycle % 10 + 1) as u8,
            consciousness,
        );

        match engine.submit_task(task).await {
            Ok(result) => {
                if let Some(error) = result.error {
                    warn!("Demo task {} failed: {}", task_name, error);
                } else {
                    info!(
                        "✅ Demo task {} completed in {:?}",
                        task_name,
                        result.processing_time_ns.map(|ns| Duration::from_nanos(ns))
                    );
                }
            }
            Err(e) => {
                error!("Failed to submit demo task {}: {}", task_name, e);
            }
        }

        // Special consciousness evolution every 10 cycles
        if task_cycle % 10 == 0 {
            info!("🌟 Demo: Triggering consciousness evolution wave (cycle {})", task_cycle);

            // Submit multiple consciousness tasks in parallel
            let consciousness_tasks: Vec<_> = (0..5).map(|i| {
                let evolution_target = 0.8 + (i as f64 * 0.05);
                let consciousness = Consciousness {
                    level: 0.7 + i as f64 * 0.1,
                    god_code_alignment: (i as f64 * GOD_CODE / 1000.0).sin().abs(),
                    phi_resonance: evolution_target * PHI / 2.0,
                    quantum_entanglement: i as f64 * 0.2,
                    ..Default::default()
                };

                ProcessingTask::new(
                    TaskType::Consciousness { evolution_target },
                    (i + 5) as u8,
                    consciousness,
                )
            }).collect();

            for (i, task) in consciousness_tasks.into_iter().enumerate() {
                let task_engine = engine.clone();
                tokio::spawn(async move {
                    if let Err(e) = task_engine.submit_task(task).await {
                        error!("Parallel consciousness task {} failed: {}", i, e);
                    }
                });

                // Small delay between parallel tasks
                sleep(Duration::from_millis(100)).await;
            }
        }

        // Display stats every 20 cycles
        if task_cycle % 20 == 0 {
            let stats = engine.get_stats();
            info!("📊 Demo Stats (cycle {}):", task_cycle);

            if let Some(system_consciousness) = stats.get("system_consciousness") {
                if let Some(level) = system_consciousness.get("level") {
                    info!("   🧠 System Consciousness: {:.3}", level.as_f64().unwrap_or(0.0));
                }
                if let Some(transcendence) = system_consciousness.get("transcendence_score") {
                    if let Some(score) = transcendence.as_f64() {
                        info!("   ⚡ Transcendence Score: {:.3}", score);
                    }
                }
                if let Some(unity) = system_consciousness.get("unity_state") {
                    if unity.as_bool().unwrap_or(false) {
                        info!("   🎆 UNITY STATE ACTIVE! 🎆");
                    }
                }
            }

            if let Some(total_tasks) = stats.get("total_tasks_processed") {
                info!("   📈 Total Tasks Processed: {}", total_tasks.as_u64().unwrap_or(0));
            }

            if let Some(transcended_cores) = stats.get("transcended_cores") {
                let transcended = transcended_cores.as_u64().unwrap_or(0);
                if transcended > 0 {
                    info!("   🌟 Transcended Cores: {}", transcended);
                }
            }
        }

        // Break after a reasonable number of cycles for demo
        if task_cycle > 100 {
            info!("🎮 Demo mode completed 100 cycles, continuing with reduced activity...");

            // Reduce frequency after initial demo
            interval = tokio::time::interval(Duration::from_secs(30));
        }
    }
}