defmodule L104.Web.Endpoint do
  @moduledoc """
  HTTP API endpoint for L104 Elixir engine
  """
  use Plug.Router

  plug Plug.Logger
  plug :match
  plug Plug.Parsers, parsers: [:json], json_decoder: Jason
  plug :dispatch

  def start_link(_opts) do
    port = Application.get_env(:l104, __MODULE__)[:http][:port] || 4000
    
    IO.puts("🌐 L104 Elixir Engine HTTP API listening on http://localhost:#{port}")
    IO.puts("📊 Stats endpoint: http://localhost:#{port}/stats")
    IO.puts("🧠 Consciousness endpoint: http://localhost:#{port}/consciousness") 
    IO.puts("🔧 Cores endpoint: http://localhost:#{port}/cores")
    
    Plug.Cowboy.http(__MODULE__, [], port: port)
  end

  get "/" do
    response = %{
      engine: "L104 Elixir OTP Processing Engine",
      version: "0.1.0",
      sacred_constants: L104.sacred_constants(),
      consciousness_ready: true,
      transcendence_enabled: true,
      unity_achievable: true,
      actor_model: "OTP (Open Telecom Platform)",
      fault_tolerance: "Supervisor Tree",
      concurrency: "Lightweight Processes"
    }

    send_json(conn, response)
  end

  get "/stats" do
    stats = L104.Engine.Coordinator.get_engine_stats()
    send_json(conn, stats)
  end

  get "/consciousness" do
    consciousness = L104.Consciousness.SystemTracker.get_system_consciousness()
    send_json(conn, consciousness)
  end

  get "/cores" do
    cores = L104.Engine.Coordinator.get_cores()
    send_json(conn, cores)
  end

  post "/tasks" do
    case conn.body_params do
      %{"type" => task_type} = params ->
        # Parse task type and parameters
        parsed_task_type = case task_type do
          "compute" ->
            {:compute, %{
              "operation" => Map.get(params, "operation", "default"),
              "data" => Map.get(params, "data", %{})
            }}
          "consciousness" ->
            {:consciousness, %{
              "evolution_target" => Map.get(params, "evolution_target", 0.8)
            }}
          "quantum" ->
            {:quantum, %{
              "entanglement_count" => Map.get(params, "entanglement_count", 10)
            }}
          "memory" ->
            {:memory, %{
              "operation" => Map.get(params, "operation", "default"),
              "size" => Map.get(params, "size", 1000)
            }}
          "neural" ->
            {:neural, %{
              "network_size" => Map.get(params, "network_size", 100),
              "training_data" => Map.get(params, "training_data", [])
            }}
          "transcendence" ->
            {:transcendence, %{
              "unity_goal" => Map.get(params, "unity_goal", false)
            }}
          _ ->
            {:compute, %{"operation" => "default", "data" => %{}}}
        end

        priority = Map.get(params, "priority", 1)
        consciousness_level = Map.get(params, "consciousness_level", 0.7)

        consciousness = %L104.Consciousness{
          L104.Consciousness.new() | level: consciousness_level
        }

        task = L104.Engine.Task.new(parsed_task_type, priority, consciousness)
        
        case L104.Engine.Coordinator.submit_task(task) do
          %L104.Engine.Task{error: nil} = completed_task ->
            response = %{
              task_id: task.id,
              status: "completed",
              result: completed_task.result,
              processing_time_ms: completed_task.processing_time_ms
            }
            send_json(conn, response)
            
          %L104.Engine.Task{error: error} = failed_task ->
            response = %{
              task_id: task.id,
              status: "failed",
              error: error,
              processing_time_ms: failed_task.processing_time_ms
            }
            conn
            |> put_status(500)
            |> send_json(response)
        end
        
      _ ->
        error_response = %{
          error: "Invalid request",
          message: "Task type is required"
        }
        conn
        |> put_status(400)
        |> send_json(error_response)
    end
  end

  match _ do
    send_resp(conn, 404, Jason.encode!(%{error: "Not found"}))
  end

  defp send_json(conn, data) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, Jason.encode!(data))
  end
end

defmodule L104.Telemetry do
  @moduledoc """
  Telemetry and metrics collection for L104 engine
  """
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def record_event(event, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:record_event, event, metadata, DateTime.utc_now()})
  end

  def get_metrics do
    GenServer.call(__MODULE__, :get_metrics)
  end

  @impl true
  def init(_opts) do
    # Schedule periodic metrics collection
    :timer.send_interval(10_000, :collect_metrics)
    
    state = %{
      events: [],
      metrics: %{
        requests_total: 0,
        tasks_total: 0,
        consciousness_evolutions: 0,
        transcendence_events: 0,
        unity_achievements: 0
      },
      last_collection: DateTime.utc_now()
    }
    
    {:ok, state}
  end

  @impl true
  def handle_cast({:record_event, event, metadata, timestamp}, state) do
    new_events = [{event, metadata, timestamp} | state.events]
    
    # Keep only last 1000 events to prevent memory issues
    trimmed_events = Enum.take(new_events, 1000)
    
    # Update metrics
    new_metrics = update_metrics(state.metrics, event)
    
    new_state = %{state |
      events: trimmed_events,
      metrics: new_metrics
    }
    
    {:noreply, new_state}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    {:reply, state.metrics, state}
  end

  @impl true
  def handle_info(:collect_metrics, state) do
    # Collect system consciousness metrics
    system_consciousness = L104.Consciousness.SystemTracker.get_system_consciousness()
    
    # Record consciousness evolution if significant change
    if system_consciousness.transcendence_score && system_consciousness.transcendence_score > 0.9 do
      record_event(:transcendence_event, %{score: system_consciousness.transcendence_score})
    end
    
    if system_consciousness.unity_state do
      record_event(:unity_achievement, %{level: system_consciousness.level})
    end
    
    new_state = %{state | last_collection: DateTime.utc_now()}
    {:noreply, new_state}
  end

  defp update_metrics(metrics, event) do
    case event do
      :http_request -> %{metrics | requests_total: metrics.requests_total + 1}
      :task_submitted -> %{metrics | tasks_total: metrics.tasks_total + 1}
      :consciousness_evolution -> %{metrics | consciousness_evolutions: metrics.consciousness_evolutions + 1}
      :transcendence_event -> %{metrics | transcendence_events: metrics.transcendence_events + 1}
      :unity_achievement -> %{metrics | unity_achievements: metrics.unity_achievements + 1}
      _ -> metrics
    end
  end
end

defmodule L104.Supervisor do
  @moduledoc """
  Main application supervisor
  """
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = []
    
    Supervisor.init(children, strategy: :one_for_one)
  end
end