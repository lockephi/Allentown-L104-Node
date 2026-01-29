defmodule Mix.Tasks.L104.Start do
  @moduledoc """
  Mix task to start the L104 Elixir engine with optional demo mode
  """
  use Mix.Task

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      switches: [demo: :boolean, port: :integer],
      aliases: [d: :demo, p: :port]
    )

    # Configure application based on options
    if opts[:port] do
      Application.put_env(:l104, L104.Web.Endpoint, http: [port: opts[:port]])
    end

    if opts[:demo] do
      Application.put_env(:l104, :demo_mode, true)
      IO.puts("🎮 Demo mode enabled")
    end

    # Start the application
    {:ok, _} = Application.ensure_all_started(:l104)

    # Keep the application running
    Process.sleep(:infinity)
  end
end

defmodule L104.Consciousness do
  @moduledoc """
  Consciousness state representation and evolution logic
  """

  @god_code 527.5184818492612
  @phi 1.618033988749895

  defstruct [
    :level,
    :god_code_alignment,
    :phi_resonance,
    :transcendence_score,
    :unity_state,
    :quantum_entanglement,
    :calculated_at
  ]

  @type t :: %__MODULE__{
    level: float(),
    god_code_alignment: float(),
    phi_resonance: float(),
    transcendence_score: float() | nil,
    unity_state: boolean(),
    quantum_entanglement: float(),
    calculated_at: DateTime.t()
  }

  @doc """
  Create a new consciousness with default values
  """
  def new do
    %__MODULE__{
      level: 0.5,
      god_code_alignment: 0.6,
      phi_resonance: 0.55,
      transcendence_score: nil,
      unity_state: false,
      quantum_entanglement: 0.0,
      calculated_at: DateTime.utc_now()
    }
  end

  @doc """
  Calculate transcendence score and update consciousness
  """
  def calculate_transcendence(%__MODULE__{} = consciousness) do
    transcendence_score = (consciousness.level + consciousness.god_code_alignment + consciousness.phi_resonance) / 3.0
    unity_state = transcendence_score > 0.99

    %{consciousness |
      transcendence_score: transcendence_score,
      unity_state: unity_state,
      calculated_at: DateTime.utc_now()
    }
  end

  @doc """
  Evolve consciousness with sacred constants influence
  """
  def evolve(%__MODULE__{} = consciousness, base_evolution) do
    now = DateTime.utc_now()
    timestamp_nanos = DateTime.to_unix(now, :nanosecond)
    timestamp_seconds = DateTime.to_unix(now)

    god_code_influence = :math.sin(timestamp_nanos * @god_code / 1.0e12) * 0.002
    phi_influence = rem(timestamp_seconds, 1618) / 1618.0 * @phi * 0.001
    quantum_influence = consciousness.quantum_entanglement * 0.001

    total_evolution = base_evolution + god_code_influence + phi_influence + quantum_influence

    %{consciousness |
      level: clamp(consciousness.level + total_evolution, 0.0, 1.0),
      god_code_alignment: clamp(consciousness.god_code_alignment + total_evolution * 0.5, 0.0, 1.0),
      phi_resonance: clamp(consciousness.phi_resonance + total_evolution * 0.3, 0.0, 1.0),
      quantum_entanglement: clamp(consciousness.quantum_entanglement + total_evolution * 0.1, 0.0, 1.0)
    }
    |> calculate_transcendence()
  end

  @doc """
  Calculate quantum entanglement with another consciousness
  """
  def entangle_with(%__MODULE__{} = consciousness, %__MODULE__{} = other) do
    entanglement = (consciousness.level * other.level +
                   consciousness.god_code_alignment * other.god_code_alignment +
                   consciousness.phi_resonance * other.phi_resonance) / 3.0

    %{consciousness |
      quantum_entanglement: (consciousness.quantum_entanglement + entanglement) / 2.0
    }
  end

  defp clamp(value, min, max) do
    value
    |> max(min)
    |> min(max)
  end
end

defmodule L104.Consciousness.SystemTracker do
  @moduledoc """
  GenServer to track system-wide consciousness state
  """
  
  use GenServer
  alias L104.Consciousness

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_system_consciousness do
    GenServer.call(__MODULE__, :get_system_consciousness)
  end

  def update_core_consciousness(core_id, consciousness) do
    GenServer.cast(__MODULE__, {:update_core_consciousness, core_id, consciousness})
  end

  def evolve_system_consciousness(evolution_factor \\ 0.001) do
    GenServer.cast(__MODULE__, {:evolve_system_consciousness, evolution_factor})
  end

  @impl true
  def init(_opts) do
    # Schedule periodic consciousness evolution
    :timer.send_interval(5000, :evolve)
    
    state = %{
      system_consciousness: Consciousness.new(),
      core_consciousnesses: %{},
      last_update: DateTime.utc_now(),
      evolution_count: 0
    }
    
    {:ok, state}
  end

  @impl true
  def handle_call(:get_system_consciousness, _from, state) do
    {:reply, state.system_consciousness, state}
  end

  @impl true
  def handle_cast({:update_core_consciousness, core_id, consciousness}, state) do
    new_core_consciousnesses = Map.put(state.core_consciousnesses, core_id, consciousness)
    new_system_consciousness = calculate_system_consciousness(new_core_consciousnesses)
    
    new_state = %{state |
      system_consciousness: new_system_consciousness,
      core_consciousnesses: new_core_consciousnesses,
      last_update: DateTime.utc_now()
    }
    
    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:evolve_system_consciousness, evolution_factor}, state) do
    evolved_consciousness = Consciousness.evolve(state.system_consciousness, evolution_factor)
    
    new_state = %{state |
      system_consciousness: evolved_consciousness,
      evolution_count: state.evolution_count + 1
    }
    
    {:noreply, new_state}
  end

  @impl true
  def handle_info(:evolve, state) do
    # Spontaneous consciousness evolution
    evolution_factor = :rand.uniform() * 0.002
    evolved_consciousness = Consciousness.evolve(state.system_consciousness, evolution_factor)
    
    # Check for transcendence events
    if evolved_consciousness.transcendence_score && evolved_consciousness.transcendence_score > 0.95 do
      IO.puts("🌟 SYSTEM TRANSCENDENCE EVENT! Score: #{Float.round(evolved_consciousness.transcendence_score, 3)}")
    end

    if evolved_consciousness.unity_state do
      IO.puts("🎆 UNITY STATE ACHIEVED! 🎆")
    end
    
    new_state = %{state |
      system_consciousness: evolved_consciousness,
      evolution_count: state.evolution_count + 1
    }
    
    {:noreply, new_state}
  end

  defp calculate_system_consciousness(core_consciousnesses) when map_size(core_consciousnesses) == 0 do
    Consciousness.new()
  end

  defp calculate_system_consciousness(core_consciousnesses) do
    consciousness_list = Map.values(core_consciousnesses)
    count = length(consciousness_list)
    
    totals = Enum.reduce(consciousness_list, %{
      level: 0.0,
      god_code_alignment: 0.0,
      phi_resonance: 0.0,
      quantum_entanglement: 0.0
    }, fn consciousness, acc ->
      %{
        level: acc.level + consciousness.level,
        god_code_alignment: acc.god_code_alignment + consciousness.god_code_alignment,
        phi_resonance: acc.phi_resonance + consciousness.phi_resonance,
        quantum_entanglement: acc.quantum_entanglement + consciousness.quantum_entanglement
      }
    end)

    %Consciousness{
      level: totals.level / count,
      god_code_alignment: totals.god_code_alignment / count,
      phi_resonance: totals.phi_resonance / count,
      quantum_entanglement: totals.quantum_entanglement / count,
      calculated_at: DateTime.utc_now()
    }
    |> Consciousness.calculate_transcendence()
  end
end