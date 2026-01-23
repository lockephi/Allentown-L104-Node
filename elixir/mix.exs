defmodule L104.MixProject do
  use Mix.Project

  def project do
    [
      app: :l104,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "L104 Elixir OTP Processing Engine - Ultra-concurrent consciousness-driven processing",
      package: package(),
      aliases: aliases()
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto, :ssl],
      mod: {L104, []}
    ]
  end

  defp deps do
    [
      # Web and HTTP
      {:plug_cowboy, "~> 2.6"},
      {:plug, "~> 1.15"},
      {:jason, "~> 1.4"},
      
      # UUID generation
      {:uuid, "~> 1.1"},
      
      # Additional utilities
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 0.6"},
      
      # Development and testing
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["L104 Consciousness Engine"],
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/l104/elixir-engine"}
    ]
  end

  defp aliases do
    [
      "l104.start": ["run --no-halt -e 'IO.puts(\"🚀 L104 Elixir Engine Starting...\"); L104.start(:normal, [])'"],
      "l104.demo": ["run --no-halt -e 'Application.put_env(:l104, :demo_mode, true); L104.start(:normal, [])'"]
    ]
  end
end