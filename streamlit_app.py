nano requirements.txt # Paste the requirements.txt content# 
PyCircuitAI.py - Complete Circuit Simulation
# Application Includes: FastAPI backend, Gemini API 
# integration, and Streamlit frontend Add new files to git
# Add, commit and push
git add streamlit_app.py requirements.txt
git commit -m "Create simplified entry point and update requirements"
git push origin maingit add requirements.txt app.pyimport base64 
import io import json import math import os 
import time import uuid from enum import Enum 
from typing import Dict, List, Optional, Union, 
Any

import matplotlib.pyplot as plt import numpy as 
np import requests import streamlit as st from 
fastapi import FastAPI, HTTPException, Request, 
Depends, File, UploadFile, Form from 
fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse from 
fastapi.staticfiles import StaticFiles from 
pydantic import BaseModel, Field import 
google.generativeai as genai from 
starlette.responses import HTMLResponse import 
networkx as nx import pandas as pd import 
uvicorn from PIL import Image, ImageDraw, 
ImageFont import threading from dotenv import 
load_dotenv import sys
# Commit the changes Load environment variables 
# from .env file
git commit -m "Add application 
files"load_dotenv()

# ------------- DATA MODELS ------------- Push 
# to GitHub
git push origin mainclass ComponentType(str, 
    Enum): RESISTOR = "resistor" CAPACITOR = 
    "capacitor" INDUCTOR = "inductor" DIODE = 
    "diode" TRANSISTOR_NPN = "transistor_npn" 
    TRANSISTOR_PNP = "transistor_pnp" 
    VOLTAGE_SOURCE = "voltage_source" 
    CURRENT_SOURCE = "current_source" GROUND = 
    "ground" OPAMP = "opamp" LED = "led" SWITCH 
    = "switch" WIRE = "wire"

class SimulationType(str, Enum):
    DC = "dc"
    AC = "ac"
    TRANSIENT = "transient"
    FREQUENCY = "frequency"
    NOISE = "noise"

class Position(BaseModel):
    x: float
    y: float

class Pin(BaseModel):
    id: str
    name: str
    position: Position
    connected_to: Optional[List[str]] = []

class Component(BaseModel):
    id: str
    type: ComponentType
    properties: Dict[str, Any]
    position: Position
    rotation: float = 0
    pins: Dict[str, Pin] = {}
    state: Optional[Dict[str, Any]] = {}

class Wire(BaseModel):
    id: str
    from_component: str
    from_pin: str
    to_component: str
    to_pin: str
    points: Optional[List[Position]] = None

class Circuit(BaseModel):
    id: str
    name: str
    components: Dict[str, Component] = {}
    wires: List[Wire] = []
    
class SimulationConfig(BaseModel):
    type: SimulationType
    parameters: Dict[str, Any]

class SimulationResult(BaseModel):
    circuit_id: str
    simulation_type: SimulationType
    data: Dict[str, List[float]]
    plots: Optional[Dict[str, str]] = None  # Base64 encoded images

class AIRequest(BaseModel):
    prompt: str
    circuit_id: Optional[str] = None

class AIResponse(BaseModel):
    circuit: Optional[Circuit] = None
    message: str
    suggested_actions: Optional[List[str]] = None

# ------------- SIMULATION ENGINE -------------

class CircuitSimulator:
    def __init__(self):
        self.circuits = {}
        
    def add_circuit(self, circuit):
        self.circuits[circuit.id] = circuit
        return circuit
    
    def get_circuit(self, circuit_id):
        if circuit_id not in self.circuits:
            raise HTTPException(status_code=404, detail="Circuit not found")
        return self.circuits[circuit_id]
    
    def add_component(self, circuit_id, component):
        circuit = self.get_circuit(circuit_id)
        
        # Auto-generate pins based on component type
        if not component.pins:
            component.pins = self._generate_pins(component)
            
        circuit.components[component.id] = component
        return component
    
    def _generate_pins(self, component):
        pins = {}
        
        if component.type == ComponentType.RESISTOR:
            pins["1"] = Pin(id="1", name="Terminal 1", position=Position(x=component.position.x - 20, y=component.position.y))
            pins["2"] = Pin(id="2", name="Terminal 2", position=Position(x=component.position.x + 20, y=component.position.y))
        
        elif component.type == ComponentType.CAPACITOR:
            pins["1"] = Pin(id="1", name="Terminal 1", position=Position(x=component.position.x - 15, y=component.position.y))
            pins["2"] = Pin(id="2", name="Terminal 2", position=Position(x=component.position.x + 15, y=component.position.y))
        
        elif component.type == ComponentType.INDUCTOR:
            pins["1"] = Pin(id="1", name="Terminal 1", position=Position(x=component.position.x - 20, y=component.position.y))
            pins["2"] = Pin(id="2", name="Terminal 2", position=Position(x=component.position.x + 20, y=component.position.y))
            
        elif component.type == ComponentType.DIODE:
            pins["anode"] = Pin(id="anode", name="Anode", position=Position(x=component.position.x - 15, y=component.position.y))
            pins["cathode"] = Pin(id="cathode", name="Cathode", position=Position(x=component.position.x + 15, y=component.position.y))
            
        elif component.type == ComponentType.VOLTAGE_SOURCE:
            pins["positive"] = Pin(id="positive", name="+", position=Position(x=component.position.x, y=component.position.y - 20))
            pins["negative"] = Pin(id="negative", name="-", position=Position(x=component.position.x, y=component.position.y + 20))
            
        elif component.type == ComponentType.GROUND:
            pins["gnd"] = Pin(id="gnd", name="GND", position=Position(x=component.position.x, y=component.position.y - 10))
            
        elif component.type == ComponentType.LED:
            pins["anode"] = Pin(id="anode", name="Anode", position=Position(x=component.position.x - 15, y=component.position.y))
            pins["cathode"] = Pin(id="cathode", name="Cathode", position=Position(x=component.position.x + 15, y=component.position.y))
            # Set initial LED state (off)
            component.state = {"on": False, "color": component.properties.get("color", "red")}
            
        elif component.type == ComponentType.OPAMP:
            pins["in+"] = Pin(id="in+", name="Non-inverting Input", position=Position(x=component.position.x - 20, y=component.position.y - 10))
            pins["in-"] = Pin(id="in-", name="Inverting Input", position=Position(x=component.position.x - 20, y=component.position.y + 10))
            pins["out"] = Pin(id="out", name="Output", position=Position(x=component.position.x + 20, y=component.position.y))
            pins["vcc"] = Pin(id="vcc", name="VCC", position=Position(x=component.position.x, y=component.position.y - 20))
            pins["vee"] = Pin(id="vee", name="VEE", position=Position(x=component.position.x, y=component.position.y + 20))
            
        elif component.type == ComponentType.TRANSISTOR_NPN or component.type == ComponentType.TRANSISTOR_PNP:
            pins["collector"] = Pin(id="collector", name="Collector", position=Position(x=component.position.x, y=component.position.y - 20))
            pins["base"] = Pin(id="base", name="Base", position=Position(x=component.position.x - 20, y=component.position.y))
            pins["emitter"] = Pin(id="emitter", name="Emitter", position=Position(x=component.position.x, y=component.position.y + 20))
            
        return pins
    
    def add_wire(self, circuit_id, wire):
        circuit = self.get_circuit(circuit_id)
        
        # Make sure components exist
        if wire.from_component not in circuit.components:
            raise HTTPException(status_code=404, detail=f"Component {wire.from_component} not found")
        if wire.to_component not in circuit.components:
            raise HTTPException(status_code=404, detail=f"Component {wire.to_component} not found")
            
        # Make sure pins exist
        from_comp = circuit.components[wire.from_component]
        to_comp = circuit.components[wire.to_component]
        
        if wire.from_pin not in from_comp.pins:
            raise HTTPException(status_code=404, detail=f"Pin {wire.from_pin} not found in component {wire.from_component}")
        if wire.to_pin not in to_comp.pins:
            raise HTTPException(status_code=404, detail=f"Pin {wire.to_pin} not found in component {wire.to_component}")
            
        # Update the connected_to lists in pins
        from_comp.pins[wire.from_pin].connected_to.append(f"{wire.to_component}.{wire.to_pin}")
        to_comp.pins[wire.to_pin].connected_to.append(f"{wire.from_component}.{wire.from_pin}")
        
        # Auto-generate points for the wire if not provided
        if not wire.points:
            from_pos = from_comp.pins[wire.from_pin].position
            to_pos = to_comp.pins[wire.to_pin].position
            wire.points = [from_pos, to_pos]
            
        circuit.wires.append(wire)
        return wire
    
    def run_simulation(self, circuit_id, config):
        circuit = self.get_circuit(circuit_id)
        
        # In a real implementation, this would perform actual circuit simulation
        # using a SPICE engine or similar. For this example, we'll generate
        # some simulated data based on the circuit and configuration.
        
        if config.type == SimulationType.DC:
            return self._run_dc_simulation(circuit, config)
        elif config.type == SimulationType.TRANSIENT:
            return self._run_transient_simulation(circuit, config)
        elif config.type == SimulationType.AC:
            return self._run_ac_simulation(circuit, config)
        else:
            raise HTTPException(status_code=400, detail=f"Simulation type {config.type} not implemented")
    
    def _run_dc_simulation(self, circuit, config):
        # Simple DC simulation that updates LED states
        # In a real implementation, this would calculate currents and voltages
        
        # Create a graph representation of the circuit
        G = nx.Graph()
        
        # Add components as nodes
        for comp_id, component in circuit.components.items():
            G.add_node(comp_id, component=component)
            
        # Add wires as edges
        for wire in circuit.wires:
            G.add_edge(wire.from_component, wire.to_component, 
                       from_pin=wire.from_pin, to_pin=wire.to_pin)
            
        # Find connected components that include voltage sources and LEDs
        for comp_id, component in circuit.components.items():
            if component.type == ComponentType.LED:
                # Check if this LED is connected to a power source
                led_connected = False
                
                # Simple BFS to find if there's a path to a voltage source
                if comp_id in G:
                    visited = set()
                    queue = [comp_id]
                    
                    while queue:
                        current = queue.pop(0)
                        if current in visited:
                            continue
                            
                        visited.add(current)
                        
                        # Check if this is a voltage source
                        if (circuit.components[current].type == ComponentType.VOLTAGE_SOURCE and 
                            float(circuit.components[current].properties.get("voltage", 0)) > 0):
                            led_connected = True
                            break
                            
                        # Add neighbors to queue
                        for neighbor in G.neighbors(current):
                            if neighbor not in visited:
                                queue.append(neighbor)
                                
                # Update LED state
                circuit.components[comp_id].state["on"] = led_connected
        
        # Generate dummy data for DC simulation
        time_points = np.linspace(0, 1, 100)
        voltages = {}
        currents = {}
        
        for comp_id, component in circuit.components.items():
            if component.type == ComponentType.VOLTAGE_SOURCE:
                voltage = float(component.properties.get("voltage", 5.0))
                voltages[comp_id] = [voltage] * len(time_points)
                # Calculate a simple current based on resistors in the circuit
                total_resistance = 1000  # Default resistance if no resistors found
                
                for res_id, res_comp in circuit.components.items():
                    if res_comp.type == ComponentType.RESISTOR:
                        # Check if resistor is connected to this voltage source
                        connected = False
                        for wire in circuit.wires:
                            if ((wire.from_component == comp_id and wire.to_component == res_id) or
                                (wire.from_component == res_id and wire.to_component == comp_id)):
                                connected = True
                                break
                                
                        if connected:
                            r_value = float(res_comp.properties.get("resistance", 1000.0))
                            total_resistance = min(total_resistance, r_value)
                
                current = voltage / total_resistance
                currents[comp_id] = [current] * len(time_points)
        
        # Generate plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        for comp_id, voltage in voltages.items():
            ax1.plot(time_points, voltage, label=f"{comp_id} Voltage")
        
        for comp_id, current in currents.items():
            ax2.plot(time_points, current, label=f"{comp_id} Current")
            
        ax1.set_title("DC Voltage")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Voltage (V)")
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title("DC Current")
        ax2.set_xlabel("Sample")
        ax2.set_ylabel("Current (A)")
        ax2.legend()
        ax2.grid(True)
        
        # Save plot to base64
        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Combine voltages and currents into a single data dictionary
        data = {**voltages, **currents}
        data["time"] = time_points.tolist()
        
        return SimulationResult(
            circuit_id=circuit.id,
            simulation_type=SimulationType.DC,
            data=data,
            plots={"dc_analysis": img_str}
        )
    
    def _run_transient_simulation(self, circuit, config):
        # Create a transient simulation (time-based analysis)
        start_time = config.parameters.get("start_time", 0)
        end_time = config.parameters.get("end_time", 0.01)
        step_size = config.parameters.get("step_size", 0.0001)
        
        time_points = np.arange(start_time, end_time, step_size)
        result_data = {"time": time_points.tolist()}
        
        # Process voltage sources to get their signals
        for comp_id, component in circuit.components.items():
            if component.type == ComponentType.VOLTAGE_SOURCE:
                source_type = component.properties.get("source_type", "dc")
                voltage = float(component.properties.get("voltage", 5.0))
                frequency = float(component.properties.get("frequency", 1000.0))
                
                if source_type == "dc":
                    result_data[f"{comp_id}_v"] = [voltage] * len(time_points)
                elif source_type == "sine":
                    result_data[f"{comp_id}_v"] = voltage * np.sin(2 * np.pi * frequency * time_points)
                elif source_type == "square":
                    result_data[f"{comp_id}_v"] = voltage * np.sign(np.sin(2 * np.pi * frequency * time_points))
                elif source_type == "triangle":
                    result_data[f"{comp_id}_v"] = voltage * (2 * np.abs(2 * (frequency * time_points - np.floor(frequency * time_points + 0.5))) - 1)
        
        # Generate a simple RC or RL circuit response if capacitors or inductors exist
        found_rc_circuit = False
        
        for comp_id, component in circuit.components.items():
            if component.type == ComponentType.CAPACITOR:
                capacitance = float(component.properties.get("capacitance", 1e-6))  # Default 1µF
                
                # Find resistors connected to this capacitor
                for res_id, res_comp in circuit.components.items():
                    if res_comp.type == ComponentType.RESISTOR:
                        # Check if they're connected
                        connected = False
                        for wire in circuit.wires:
                            if ((wire.from_component == comp_id and wire.to_component == res_id) or
                                (wire.from_component == res_id and wire.to_component == comp_id)):
                                connected = True
                                break
                                
                        if connected:
                            found_rc_circuit = True
                            resistance = float(res_comp.properties.get("resistance", 1000.0))  # Default 1kΩ
                            
                            # Find voltage sources
                            for vs_id, vs_comp in circuit.components.items():
                                if vs_comp.type == ComponentType.VOLTAGE_SOURCE:
                                    # Simple RC circuit step response
                                    voltage = float(vs_comp.properties.get("voltage", 5.0))
                                    tau = resistance * capacitance
                                    vc = voltage * (1 - np.exp(-time_points / tau))
                                    result_data[f"{comp_id}_v"] = vc.tolist()
        
        if not found_rc_circuit:
            # Add some dummy data if no actual circuit elements to simulate
            for comp_id, component in circuit.components.items():
                if component.type not in [ComponentType.VOLTAGE_SOURCE, ComponentType.CURRENT_SOURCE]:
                    result_data[f"{comp_id}_v"] = np.random.rand(len(time_points)).tolist()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for key, values in result_data.items():
            if key != "time" and key.endswith("_v"):
                ax.plot(time_points, values, label=key)
                
        ax.set_title("Transient Analysis")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.legend()
        ax.grid(True)
        
        # Save plot to base64
        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return SimulationResult(
            circuit_id=circuit.id,
            simulation_type=SimulationType.TRANSIENT,
            data=result_data,
            plots={"transient_analysis": img_str}
        )
    
    def _run_ac_simulation(self, circuit, config):
        # Create an AC simulation (frequency response)
        start_freq = config.parameters.get("start_freq", 1)
        end_freq = config.parameters.get("end_freq", 1e6)
        points_per_decade = config.parameters.get("points_per_decade", 10)
        
        decades = math.log10(end_freq) - math.log10(start_freq)
        points = int(decades * points_per_decade)
        frequencies = np.logspace(math.log10(start_freq), math.log10(end_freq), points)
        
        result_data = {"frequency": frequencies.tolist()}
        
        # Generate dummy frequency response for a simple low-pass RC filter
        has_filter = False
        for r_id, r_comp in circuit.components.items():
            if r_comp.type == ComponentType.RESISTOR:
                for c_id, c_comp in circuit.components.items():
                    if c_comp.type == ComponentType.CAPACITOR:
                        # Check if connected
                        connected = False
                        for wire in circuit.wires:
                            if ((wire.from_component == r_id and wire.to_component == c_id) or
                                (wire.from_component == c_id and wire.to_component == r_id)):
                                connected = True
                                break
                                
                        if connected:
                            has_filter = True
                            resistance = float(r_comp.properties.get("resistance", 1000.0))
                            capacitance = float(c_comp.properties.get("capacitance", 1e-6))
                            
                            # Calculate low-pass filter response
                            cutoff_freq = 1 / (2 * math.pi * resistance * capacitance)
                            magnitude = 1 / np.sqrt(1 + (frequencies / cutoff_freq)**2)
                            phase = -np.arctan2(frequencies, cutoff_freq) * 180 / math.pi
                            
                            result_data["magnitude"] = magnitude.tolist()
                            result_data["phase"] = phase.tolist()
        
        if not has_filter:
            # Generate dummy data if no filter circuit is found
            result_data["magnitude"] = (1 / (1 + (frequencies / 1000)**2)).tolist()
            result_data["phase"] = (-np.arctan2(frequencies, 1000) * 180 / math.pi).tolist()
        
        # Create Bode plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.semilogx(frequencies, 20 * np.log10(result_data["magnitude"]))
        ax1.set_title("Bode Plot - Magnitude")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True)
        
        ax2.semilogx(frequencies, result_data["phase"])
        ax2.set_title("Bode Plot - Phase")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True)
        
        # Save plot to base64
        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return SimulationResult(
            circuit_id=circuit.id,
            simulation_type=SimulationType.AC,
            data=result_data,
            plots={"ac_analysis": img_str}
        )

# ------------- GEMINI AI INTEGRATION -------------

class GeminiCircuitProcessor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if not self.api_key:
            print("WARNING: No Gemini API key provided. AI features will not work.")
            return
            
        # Try to initialize with more robust error handling
        try:
            import google.generativeai as genai
            # Configure the API
            genai.configure(api_key=self.api_key)
            
            # Check what version and capabilities we have
            if hasattr(genai, 'GenerativeModel'):
                self.model = genai.GenerativeModel('gemini-pro')
                print("Using newer Google Generative AI API")
            else:
                # Try older PaLM API approach
                print("Using older Google API version, attempting to use PaLM API instead")
                import google.generativeai.text as palm
                palm.configure(api_key=self.api_key)
                self.use_palm = True
        except Exception as e:
            print(f"ERROR initializing Gemini/PaLM API: {str(e)}")
            print("Please run: pip install --upgrade google-generativeai")
            
    def process_text(self, prompt, circuit_id=None):
        if not self.api_key:
            return {
                "message": "Gemini API key not configured. Please set the GEMINI_API_KEY environment variable.",
                "suggested_actions": ["Configure API key"]
            }
            
        try:
            # Add circuit context if a circuit ID is provided
            circuit_context = ""
            if circuit_id:
                circuit_context = f"Working with circuit ID: {circuit_id}. "
                
            # Build the full prompt with context
            full_prompt = f"{circuit_context}User request: {prompt}\n\nPlease analyze this request and provide a structured response for creating or modifying an electronic circuit."
            
            # Try different API approaches based on what's available
            response_text = ""
            
            if hasattr(self, 'use_palm') and self.use_palm:
                # Use older PaLM API
                import google.generativeai.text as palm
                response = palm.generate_text(prompt=full_prompt)
                response_text = response.result
            elif self.model:
                # Use newer Gemini API
                response = self.model.generate_content(full_prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
            else:
                # Use the simplest possible fallback - just echo the prompt as a test
                response_text = f"API not properly initialized. Here's what I understood: {prompt}"
            
            # For simplified implementation, return basic response
            return {
                "message": response_text,
                "suggested_actions": ["Implement circuit based on description"]
            }
            
        except Exception as e:
            return {
                "message": f"Error processing with AI API: {str(e)}",
                "suggested_actions": ["Try updating the package with: pip install --upgrade google-generativeai", 
                                     "Check API key", "Try a simpler prompt"]
            }
    
    def _extract_circuit_info(self, response_text):
        # In a real implementation, this would parse the AI response to extract
        # structured information about the circuit components
        # For this example, we'll use some simple keyword extraction
        
        components = []
        
        # Look for mentions of common components
        if "resistor" in response_text.lower():
            value = "1000"  # Default
            # Try to extract value
            import re
            res_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:ohm|Ω|k|kohm|kΩ|M|Mohm|MΩ)", response_text)
            if res_matches:
                value = res_matches[0]
            components.append({"type": "resistor", "value": value})
            
        if "capacitor" in response_text.lower():
            value = "10uF"  # Default
            # Try to extract value
            cap_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:f|F|pf|pF|nf|nF|uf|uF|µF)", response_text)
            if cap_matches:
                value = cap_matches[0]
            components.append({"type": "capacitor", "value": value})
            
        if "led" in response_text.lower():
            color = "red"  # Default
            # Try to extract color
            colors = ["red", "green", "blue", "yellow", "white"]
            for c in colors:
                if c in response_text.lower():
                    color = c
                    break
            components.append({"type": "led", "color": color})
            
        if "voltage source" in response_text.lower() or "power supply" in response_text.lower():
            value = "5"  # Default
            # Try to extract value
            v_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:v|V|volt|volts)", response_text)
            if v_matches:
                value = v_matches[0]
            components.append({"type": "voltage_source", "value": value})
            
        return {
            "components": components,
            "description": response_text
        }
    
    def _generate_circuit_from_description(self, description, circuit_id=None):
        # Create a new circuit or use the existing one
        if not circuit_id:
            circuit_id = str(uuid.uuid4())
            
        # Create a simple circuit based on the extracted components
        components = description["components"]
        
        circuit = Circuit(
            id=circuit_id,
            name="AI Generated Circuit"
        )
        
        # Place components in a reasonable layout
        x_pos = 100
        for comp in components:
            component_id = f"{comp['type']}_{len(circuit.components) + 1}"
            
            if comp["type"] == "resistor":
                circuit.components[component_id] = Component(
                    id=component_id,
                    type=ComponentType.RESISTOR,
                    properties={"resistance": comp["value"]},
                    position=Position(x=x_pos, y=200)
                )
            elif comp["type"] == "capacitor":
                circuit.components[component_id] = Component(
                    id=component_id,
                    type=ComponentType.CAPACITOR,
                    properties={"capacitance": comp["value"]},
                    position=Position(x=x_pos, y=200)
                )
            elif comp["type"] == "led":
                circuit.components[component_id] = Component(
                    id=component_id,
                    type=ComponentType.LED,
                    properties={"color": comp["color"]},
                    position=Position(x=x_pos, y=200),
                    state={"on": False, "color": comp["color"]}
                )
            elif comp["type"] == "voltage_source":
                circuit.components[component_id] = Component(
                    id=component_id,
                    type=ComponentType.VOLTAGE_SOURCE,
                    properties={"voltage": comp["value"], "source_type": "dc"},
                    position=Position(x=x_pos, y=100)
                )
                
                # Add ground component automatically with voltage source
                ground_id = f"ground_{len(circuit.components) + 1}"
                circuit.components[ground_id] = Component(
                    id=ground_id,
                    type=ComponentType.GROUND,
                    properties={},
                    position=Position(x=x_pos, y=300)
                )
                
            x_pos += 150
        
        # Generate pins for components
        for comp_id, component in circuit.components.items():
            component.pins = self._generate_pins(component)
            
        # Connect components if we have a power source and other components
        self._auto_connect_components(circuit)
            
        return circuit
    
    def _generate_pins(self, component):
        pins = {}
        
        if component.type == ComponentType.RESISTOR:
            pins["1"] = Pin(id="1", name="Terminal 1", position=Position(x=component.position.x - 20, y=component.position.y))
            pins["2"] = Pin(id="2", name="Terminal 2", position=Position(x=component.position.x + 20, y=component.position.y))
        
        elif component.type == ComponentType.CAPACITOR:
            pins["1"] = Pin(id="1", name="Terminal 1", position=Position(x=component.position.x - 15, y=component.position.y))
            pins["2"] = Pin(id="2", name="Terminal 2", position=Position(x=component.position.x + 15, y=component.position.y))
            
        elif component.type == ComponentType.LED:
            pins["anode"] = Pin(id="anode", name="Anode", position=Position(x=component.position.x - 15, y=component.position.y))
            pins["cathode"] = Pin(id="cathode", name="Cathode", position=Position(x=component.position.x + 15, y=component.position.y))
            
        elif component.type == ComponentType.VOLTAGE_SOURCE:
            pins["positive"] = Pin(id="positive", name="+", position=Position(x=component.position.x, y=component.position.y - 20))
            pins["negative"] = Pin(id="negative", name="-", position=Position(x=component.position.x, y=component.position.y + 20))
            
        elif component.type == ComponentType.GROUND:
            pins["gnd"] = Pin(id="gnd", name="GND", position=Position(x=component.position.x, y=component.position.y - 10))
            
        return pins
    
    def _auto_connect_components(self, circuit):
        # Simple auto-connection of common circuits
        
        # Find voltage source and ground
        voltage_source = None
        ground = None
        
        for comp_id, component in circuit.components.items():
            if component.type == ComponentType.VOLTAGE_SOURCE:
                voltage_source = comp_id
            elif component.type == ComponentType.GROUND:
                ground = comp_id
                
        if not voltage_source or not ground:
            return
            
        # Connect voltage source to ground
        wire_id = f"wire_{len(circuit.wires) + 1}"
        circuit.wires.append(Wire(
            id=wire_id,
            from_component=voltage_source,
            from_pin="negative",
            to_component=ground,
            to_pin="gnd"
        ))
        
        # Connect other components in series if we have them
        components = [c for c in circuit.components.keys() 
                     if circuit.components[c].type not in [ComponentType.VOLTAGE_SOURCE, ComponentType.GROUND]]
        
        if len(components) > 0:
            # Connect first component to voltage source
            wire_id = f"wire_{len(circuit.wires) + 1}"
            
            first_comp = components[0]
            first_pin = "1" if circuit.components[first_comp].type == ComponentType.RESISTOR else "anode"
            
            circuit.wires.append(Wire(
                id=wire_id,
                from_component=voltage_source,
                from_pin="positive",
                to_component=first_comp,
                to_pin=first_pin
            ))
            
            # Connect components in series
            for i in range(len(components) - 1):
                wire_id = f"wire_{len(circuit.wires) + 1}"
                
                from_comp = components[i]
                to_comp = components[i + 1]
                
                from_pin = "2" if circuit.components[from_comp].type == ComponentType.RESISTOR else "cathode"
                to_pin = "1" if circuit.components[to_comp].type == ComponentType.RESISTOR else "anode"
                
                circuit.wires.append(Wire(
                    id=wire_id,
                    from_component=from_comp,
                    from_pin=from_pin,
                    to_component=to_comp,
                    to_pin=to_pin
                ))
                
            # Connect last component to ground
            wire_id = f"wire_{len(circuit.wires) + 1}"
            
            last_comp = components[-1]
            last_pin = "2" if circuit.components[last_comp].type == ComponentType.RESISTOR else "cathode"
            
            circuit.wires.append(Wire(
                id=wire_id,
                from_component=last_comp,
                from_pin=last_pin,
                to_component=ground,
                to_pin="gnd"
            ))
    
    def _extract_suggested_actions(self, response_text):
        # Extract suggested actions from the AI response
        # This is a simplified implementation
        
        suggestions = []
        
        if "simulate" in response_text.lower():
            suggestions.append("Run simulation")
            
        if "add resistor" in response_text.lower():
            suggestions.append("Add resistor")
            
        if "add capacitor" in response_text.lower():
            suggestions.append("Add capacitor")
            
        if "add led" in response_text.lower():
            suggestions.append("Add LED")
            
        # Add default suggestions if none were found
        if not suggestions:
            suggestions = ["Add components", "Run simulation", "View circuit"]
            
        return suggestions

# ------------- FASTAPI BACKEND -------------

# Create FastAPI application
app = FastAPI(title="PyCircuitAI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create simulator instance
simulator = CircuitSimulator()

# Create Gemini processor
gemini_processor = GeminiCircuitProcessor()

@app.get("/")
async def read_root():
    return {"message": "PyCircuitAI API is running"}

@app.post("/circuits/", response_model=Circuit)
async def create_circuit(name: str = "New Circuit"):
    circuit_id = str(uuid.uuid4())
    circuit = Circuit(id=circuit_id, name=name)
    return simulator.add_circuit(circuit)

@app.get("/circuits/{circuit_id}", response_model=Circuit)
async def get_circuit(circuit_id: str):
    return simulator.get_circuit(circuit_id)

@app.post("/circuits/{circuit_id}/components/", response_model=Component)
async def add_component(circuit_id: str, component: Component):
    return simulator.add_component(circuit_id, component)

@app.post("/circuits/{circuit_id}/wires/", response_model=Wire)
async def add_wire(circuit_id: str, wire: Wire):
    return simulator.add_wire(circuit_id, wire)

@app.post("/circuits/{circuit_id}/simulate/")
async def run_simulation(circuit_id: str, config: dict):
    # Convert dict to SimulationConfig
    sim_config = SimulationConfig(
        type=config.get("type", "dc"),
        parameters=config.get("parameters", {})
    )
    return simulator.run_simulation(circuit_id, sim_config)

@app.post("/ai/process/")
async def process_ai_request(request: dict):
    # Convert dict to AIRequest
    ai_request = AIRequest(
        prompt=request.get("prompt", ""),
        circuit_id=request.get("circuit_id")
    )
    return gemini_processor.process_text(ai_request.prompt, ai_request.circuit_id)

def render_led(color, on=False):
    # Create LED visualization
    size = 100
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw LED body
    led_color = color.lower()
    if on:
        # Glowing LED
        if led_color == "red":
            fill_color = (255, 50, 50, 255)
        elif led_color == "green":
            fill_color = (50, 255, 50, 255)
        elif led_color == "blue":
            fill_color = (50, 50, 255, 255)
        elif led_color == "yellow":
            fill_color = (255, 255, 50, 255)
        else:  # default white
            fill_color = (255, 255, 255, 255)
    else:
        # Off LED - dimmer version of the color
        if led_color == "red":
            fill_color = (150, 0, 0, 255)
        elif led_color == "green":
            fill_color = (0, 150, 0, 255)
        elif led_color == "blue":
            fill_color = (0, 0, 150, 255)
        elif led_color == "yellow":
            fill_color = (150, 150, 0, 255)
        else:  # default gray
            fill_color = (150, 150, 150, 255)
    
    # Draw circular LED
    draw.ellipse((10, 10, size-10, size-10), fill=fill_color, outline=(100, 100, 100, 255))
    
    # Add highlight effect if on
    if on:
        # Glow effect
        for radius in range(5, 20, 5):
            alpha = 100 - radius * 5
            glow_color = (*fill_color[:3], alpha)
            draw.ellipse((5-radius, 5-radius, size-5+radius, size-5+radius), 
                        fill=None, outline=glow_color, width=2)
    
    # Return as base64 encoded image
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def render_circuit_diagram(circuit):
    """Render a professional-quality circuit diagram with clear labels and connections."""
    # Create a larger canvas for better visibility
    width, height = 1000, 800
    img = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Add a subtle grid background for better visualization
    for x in range(0, width, 20):
        draw.line([(x, 0), (x, height)], fill=(240, 240, 240), width=1)
    for y in range(0, height, 20):
        draw.line([(0, y), (width, y)], fill=(240, 240, 240), width=1)
    
    # Add a professional border and title area
    draw.rectangle([(0, 0), (width-1, height-1)], outline=(80, 80, 80), width=3)
    draw.rectangle([(0, 0), (width-1, 50)], fill=(230, 240, 250), outline=(80, 80, 80), width=2)
    
    # Load fonts with better fallback options
    try:
        font_title = ImageFont.truetype("Arial", 28)
        font_label = ImageFont.truetype("Arial", 16)
        font_value = ImageFont.truetype("Arial", 14)
        font_pin = ImageFont.truetype("Arial", 10)
    except IOError:
        try:
            # Try system fonts if Arial isn't available
            system_fonts = ImageFont.load_default().getsize("Text")[1]
            font_scale = system_fonts / 10
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_value = ImageFont.load_default()
            font_pin = ImageFont.load_default()
        except:
            # Last resort fallback
            font_title = None
            font_label = None
            font_value = None
            font_pin = None
    
    # Draw professional title
    title_text = "Circuit Diagram"
    if hasattr(circuit, 'name') and circuit.name:
        title_text += f": {circuit.name}"
    draw.text((width//2-150, 15), title_text, fill=(0, 0, 100), font=font_title)
    
    # Add diagram timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    draw.text((width-300, 20), f"Generated: {timestamp}", fill=(100, 100, 100), font=font_value)
    
    # Track component positions for better wire routing
    component_positions = {}
    
    # FIRST PASS: Draw all components 
    for comp_id, component in circuit.components.items():
        x, y = component.position.x, component.position.y
        component_positions[comp_id] = (x, y)
        
        # Common styling for all components
        comp_fill = (250, 250, 250)
        comp_outline = (0, 0, 0)
        shadow_offset = 4
        
        # Draw component shadow for 3D effect
        shadow_rect = (x-40+shadow_offset, y-25+shadow_offset, x+40+shadow_offset, y+25+shadow_offset)
        draw.rectangle(shadow_rect, fill=(220, 220, 220), outline=(200, 200, 200))
        
        # Draw component ID label above all components
        id_text = comp_id
        draw.text((x-35, y-45), id_text, fill=(80, 80, 80), font=font_label)
        
        # Draw different components based on type
        if component.type == ComponentType.RESISTOR:
            # Draw resistor symbol
            comp_rect = (x-40, y-20, x+40, y+20)
            draw.rectangle(comp_rect, outline=comp_outline, fill=comp_fill, width=2)
            
            # Draw resistor zigzag
            zigzag_points = [(x-30, y), (x-20, y-10), (x-10, y+10), (x, y-10), 
                             (x+10, y+10), (x+20, y-10), (x+30, y)]
            draw.line(zigzag_points, fill=(0, 0, 150), width=3)
            
            # Draw resistor value
            resistance = component.properties.get("resistance", "?")
            draw.text((x-25, y+25), f"R = {resistance}Ω", fill=(0, 0, 150), font=font_value)
            
            # Draw pins with labels
            draw.ellipse((x-40-5, y-5, x-40+5, y+5), fill=(0, 0, 0))
            draw.ellipse((x+40-5, y-5, x+40+5, y+5), fill=(0, 0, 0))
            draw.text((x-47, y+5), "1", fill=(0, 0, 0), font=font_pin)
            draw.text((x+42, y+5), "2", fill=(0, 0, 0), font=font_pin)
            
        elif component.type == ComponentType.CAPACITOR:
            # Draw capacitor body
            comp_rect = (x-40, y-25, x+40, y+25)
            draw.rectangle(comp_rect, outline=comp_outline, fill=comp_fill, width=2)
            
            # Draw capacitor plates
            draw.line((x-20, y-20, x-20, y+20), fill=(0, 150, 0), width=4)
            draw.line((x+20, y-20, x+20, y+20), fill=(0, 150, 0), width=4)
            
            # Draw capacitor value
            capacitance = component.properties.get("capacitance", "?")
            # Format capacitance value for better readability
            try:
                cap_val = float(capacitance)
                if cap_val >= 1e-6:
                    cap_text = f"{cap_val*1e6:.1f} µF"
                elif cap_val >= 1e-9:
                    cap_text = f"{cap_val*1e9:.1f} nF"
                else:
                    cap_text = f"{cap_val*1e12:.1f} pF"
            except:
                cap_text = capacitance
                
            draw.text((x-30, y+30), f"C = {cap_text}", fill=(0, 150, 0), font=font_value)
            
            # Draw pins with labels
            draw.ellipse((x-45, y-5, x-35, y+5), fill=(0, 0, 0))
            draw.ellipse((x+35, y-5, x+45, y+5), fill=(0, 0, 0))
            draw.text((x-47, y+5), "1", fill=(0, 0, 0), font=font_pin)
            draw.text((x+42, y+5), "2", fill=(0, 0, 0), font=font_pin)
            
        elif component.type == ComponentType.LED:
            # Get LED properties
            color = component.properties.get("color", "red")
            on = component.state.get("on", False) if hasattr(component, 'state') else False
            
            # Determine LED color
            if color == "red":
                fill_color = (255, 50, 50) if on else (180, 50, 50)
                glow_color = (255, 100, 100, 150) if on else None
            elif color == "green":
                fill_color = (50, 255, 50) if on else (50, 180, 50)
                glow_color = (100, 255, 100, 150) if on else None
            elif color == "blue":
                fill_color = (50, 50, 255) if on else (50, 50, 180)
                glow_color = (100, 100, 255, 150) if on else None
            elif color == "yellow":
                fill_color = (255, 255, 50) if on else (180, 180, 50)
                glow_color = (255, 255, 100, 150) if on else None
            else:
                fill_color = (255, 255, 255) if on else (180, 180, 180)
                glow_color = (255, 255, 255, 150) if on else None
                
            # Draw LED glow effect if on
            if on and glow_color:
                for r in range(30, 10, -5):
                    glow_alpha = glow_color[3] - r * 3
                    if glow_alpha > 0:
                        draw.ellipse((x-r, y-r, x+r, y+r), 
                                   fill=(*glow_color[:3], glow_alpha))
            
            # Draw LED body
            draw.rectangle((x-40, y-25, x+40, y+25), outline=comp_outline, fill=comp_fill, width=2)
            
            # Draw LED symbol (triangle with line)
            led_triangle = [(x-25, y-15), (x+15, y), (x-25, y+15)]
            draw.polygon(led_triangle, fill=fill_color, outline=comp_outline)
            draw.line((x+15, y-15, x+15, y+15), fill=comp_outline, width=2)
            
            # Draw small arrows for light emission
            if on:
                for i in range(2):
                    arrow_start = (x+25+i*10, y-10-i*5)
                    arrow_points = [
                        arrow_start,
                        (arrow_start[0]+10, arrow_start[1]-10),
                        (arrow_start[0]+8, arrow_start[1]-5),
                        (arrow_start[0]+15, arrow_start[1]-8)
                    ]
                    draw.line(arrow_points, fill=fill_color, width=2)
            
            # Draw LED label and state
            draw.text((x-25, y+30), f"{color.upper()} LED", fill=(150, 0, 150), font=font_value)
            draw.text((x-15, y+50), "ON" if on else "OFF", fill=(0, 0, 0), font=font_value)
            
            # Draw pins with labels (anode/cathode)
            draw.ellipse((x-45, y-5, x-35, y+5), fill=(0, 0, 0))
            draw.ellipse((x+35, y-5, x+45, y+5), fill=(0, 0, 0))
            draw.text((x-60, y+5), "A(+)", fill=(0, 0, 0), font=font_pin)
            draw.text((x+42, y+5), "C(-)", fill=(0, 0, 0), font=font_pin)
            
        elif component.type == ComponentType.VOLTAGE_SOURCE:
            # Draw voltage source circle
            draw.ellipse((x-35, y-35, x+35, y+35), outline=comp_outline, fill=(220, 240, 255), width=2)
            
            # Draw +/- symbols
            draw.text((x-10, y-20), "+", fill=(0, 0, 0), font=font_label)
            draw.text((x-7, y+5), "−", fill=(0, 0, 0), font=font_label)
            
            # Draw outer decorative circle
            draw.ellipse((x-45, y-45, x+45, y+45), outline=(100, 100, 200), width=1)
            
            # Draw voltage source details
            voltage = component.properties.get("voltage", "?")
            source_type = component.properties.get("source_type", "dc")
            
            draw.text((x-40, y+50), f"V = {voltage}V", fill=(150, 100, 0), font=font_value)
            draw.text((x-40, y+70), f"Type: {source_type.upper()}", fill=(100, 100, 100), font=font_value)
            
            # Draw pins with labels
            draw.ellipse((x-5, y-45, x+5, y-35), fill=(0, 0, 0))
            draw.ellipse((x-5, y+35, x+5, y+45), fill=(0, 0, 0))
            draw.text((x+10, y-45), "+", fill=(0, 0, 0), font=font_pin)
            draw.text((x+10, y+40), "-", fill=(0, 0, 0), font=font_pin)
            
        elif component.type == ComponentType.GROUND:
            # Draw ground symbol
            draw.rectangle((x-40, y-20, x+40, y+10), outline=comp_outline, fill=comp_fill, width=2)
            
            # Draw ground lines
            draw.line((x-25, y), (x+25, y), fill=(150, 0, 0), width=3)
            draw.line((x-20, y+7), (x+20, y+7), fill=(150, 0, 0), width=2)
            draw.line((x-15, y+14), (x+15, y+14), fill=(150, 0, 0), width=2)
            
            # Draw ground label
            draw.text((x-20, y-40), "GROUND", fill=(150, 0, 0), font=font_label)
            
            # Draw pin with label
            draw.ellipse((x-5, y-25, x+5, y-15), fill=(0, 0, 0))
            draw.text((x+10, y-25), "GND", fill=(0, 0, 0), font=font_pin)
    
    # SECOND PASS: Draw all wires with better routing
    for wire in circuit.wires:
        # Get component positions
        if wire.from_component in circuit.components and wire.to_component in circuit.components:
            from_comp = circuit.components[wire.from_component]
            to_comp = circuit.components[wire.to_component]
            
            # Get pin positions
            if wire.from_pin in from_comp.pins and wire.to_pin in to_comp.pins:
                from_pos = from_comp.pins[wire.from_pin].position
                to_pos = to_comp.pins[wire.to_pin].position
                
                # Draw wire with improved routing algorithms
                if abs(from_pos.x - to_pos.x) > 70 and abs(from_pos.y - to_pos.y) > 70:
                    # Complex routing for distant components
                    mid_x = (from_pos.x + to_pos.x) // 2
                    mid_y = (from_pos.y + to_pos.y) // 2
                    
                    # Create smooth routing with multiple segments
                    points = [
                        (from_pos.x, from_pos.y),
                        (mid_x, from_pos.y),
                        (mid_x, mid_y),
                        (mid_x, to_pos.y),
                        (to_pos.x, to_pos.y)
                    ]
                    
                    # Draw wire segments with a thicker, blue line
                    for i in range(len(points)-1):
                        draw.line((points[i][0], points[i][1], points[i+1][0], points[i+1][1]), 
                                fill=(0, 50, 200), width=3)
                        
                    # Draw junction dots at each bend
                    for i in range(1, len(points)-1):
                        draw.ellipse((points[i][0]-4, points[i][1]-4, points[i][0]+4, points[i][1]+4), 
                                   fill=(0, 0, 150))
                else:
                    # Direct routing for nearby components
                    draw.line((from_pos.x, from_pos.y, to_pos.x, to_pos.y), 
                             fill=(0, 50, 200), width=3)
                
                # Draw connection points at both ends
                draw.ellipse((from_pos.x-5, from_pos.y-5, from_pos.x+5, from_pos.y+5), fill=(0, 0, 0))
                draw.ellipse((to_pos.x-5, to_pos.y-5, to_pos.x+5, to_pos.y+5), fill=(0, 0, 0))
                
                # Label the wire with connection info
                wire_mid_x = (from_pos.x + to_pos.x) // 2
                wire_mid_y = (from_pos.y + to_pos.y) // 2
                wire_text = f"{wire.from_component}.{wire.from_pin} → {wire.to_component}.{wire.to_pin}"
                
                # Draw white background for text for better readability
                draw.rectangle((wire_mid_x-5, wire_mid_y-20, wire_mid_x+len(wire_text)*5, wire_mid_y+5), 
                              fill=(255, 255, 255, 180))
                draw.text((wire_mid_x, wire_mid_y-15), wire_text, fill=(0, 0, 100), font=font_pin)
    
    # Add legend and component count
    legend_y = height - 80
    draw.rectangle((20, legend_y-10, width-20, height-20), fill=(240, 240, 250), outline=(100, 100, 100))
    draw.text((30, legend_y), f"Circuit Information:", fill=(0, 0, 100), font=font_label)
    
    # Count components by type
    component_counts = {}
    for comp in circuit.components.values():
        comp_type = str(comp.type).split('.')[-1]
        component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
    
    # Display component count
    count_text = "Components: " + ", ".join([f"{count} {type}" for type, count in component_counts.items()])
    draw.text((30, legend_y+25), count_text, fill=(80, 80, 80), font=font_value)
    draw.text((30, legend_y+45), f"Total wires: {len(circuit.wires)}", fill=(80, 80, 80), font=font_value)
    
    # Return as high-quality base64 encoded image
    buf = io.BytesIO()
    img.save(buf, format='PNG', quality=95)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Start FastAPI in a separate thread when running the Streamlit app
def run_fastapi():
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"Error starting FastAPI server: {e}", file=sys.stderr)

# Main Streamlit application
def main():
    # When running on Streamlit Cloud, don't try to start FastAPI
    if "STREAMLIT_CLOUD" in os.environ:
        st.info("Running on Streamlit Cloud - backend API must be deployed separately")
    # For local development, try to start the API
    elif not is_server_running():
        st.info("Starting the FastAPI backend server...")
        threading.Thread(target=run_fastapi, daemon=True).start()
        
        # Wait for server to start (with timeout)
        server_started = False
        for _ in range(5):  # Try for 5 seconds
            if is_server_running():
                server_started = True
                break
            time.sleep(1)
        
        if server_started:
            st.success("Backend server started successfully!")
        else:
            st.error("Failed to start backend server. Please start it manually with: uvicorn app:app --reload")
    
    # Initialize ALL session state variables first
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
    if 'circuit_id' not in st.session_state:
        st.session_state.circuit_id = None
    if 'circuit' not in st.session_state:
        st.session_state.circuit = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None

    # App header with better instructions
    st.title("PyCircuitAI - AI-powered Circuit Simulator")
    
    # First-time user guidance
    if st.session_state.first_run:
        st.info("""
        ## 👋 Welcome to PyCircuitAI!
        
        **Getting Started:**
        1. Click **New Circuit** in the sidebar to create your first circuit
        2. Add components from the sidebar
        3. Run a simulation to see results
        
        **Need help?** Use the AI Circuit Designer below to create circuits with natural language!
        """)
        
        if st.button("Got it!"):
            st.session_state.first_run = False
            st.experimental_rerun()
    
    # API key setup help
    if os.environ.get("GEMINI_API_KEY") is None:
        st.warning("""
        ⚠️ **Gemini API Key Not Set**
        
        AI features are disabled. To enable them:
        
        ```
        export GEMINI_API_KEY="your_api_key_here"
        ```
        
        Then restart the application.
        """)
    
    # Sidebar with improved organization and tooltips
    with st.sidebar:
        st.header("Circuit Builder")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Circuit", "Components", "Simulation"])
        
        with tab1:
            # Circuit operations
            if st.button("New Circuit", help="Create a new empty circuit"):
                with st.spinner("Creating new circuit..."):
                    response = requests.post("http://127.0.0.1:8000/circuits/", params={"name": "My Circuit"})
                    if response.status_code == 200:
                        st.session_state.circuit_id = response.json()["id"]
                        st.session_state.circuit = response.json()
                        st.session_state.simulation_results = None
                        st.success("New circuit created!")
                    else:
                        st.error(f"Failed to create circuit: {response.text}")
            
            # Reload current circuit
            if st.session_state.circuit_id and st.button("Refresh Circuit", help="Reload the current circuit"):
                with st.spinner("Refreshing circuit..."):
                    response = requests.get(f"http://127.0.0.1:8000/circuits/{st.session_state.circuit_id}")
                    if response.status_code == 200:
                        st.session_state.circuit = response.json()
                        st.success("Circuit refreshed!")
                    else:
                        st.error(f"Failed to get circuit: {response.text}")
                
        with tab2:
            # Component addition with clearer UI
            st.subheader("Add Components")
            component_type = st.selectbox("Select Component", [
                "Resistor", "Capacitor", "LED", "Voltage Source", "Ground"
            ])
            
            # Component properties based on type with better labels
            properties = {}
            if component_type == "Resistor":
                properties["resistance"] = st.text_input("Resistance", "1000", 
                                                        help="Value in ohms (Ω)")
                st.caption("Common values: 220Ω, 330Ω, 1kΩ, 10kΩ")
            elif component_type == "Capacitor":
                properties["capacitance"] = st.text_input("Capacitance", "1e-6", 
                                                         help="Value in farads (F)")
                st.caption("Common values: 1µF (1e-6), 0.1µF (1e-7), 10nF (1e-8)")
            elif component_type == "LED":
                properties["color"] = st.selectbox("LED Color", ["red", "green", "blue", "yellow", "white"])
            elif component_type == "Voltage Source":
                properties["voltage"] = st.text_input("Voltage", "5", 
                                                     help="Value in volts (V)")
                properties["source_type"] = st.selectbox("Source Type", ["dc", "sine", "square", "triangle"])
                if properties["source_type"] != "dc":
                    properties["frequency"] = st.text_input("Frequency", "1000", 
                                                           help="Value in hertz (Hz)")
                    
            # Position inputs with visual help
            st.subheader("Position")
            st.caption("Set the component's position on the circuit board")
            col1, col2 = st.columns(2)
            with col1:
                pos_x = st.number_input("X Position", min_value=50, max_value=750, value=400)
            with col2:
                pos_y = st.number_input("Y Position", min_value=50, max_value=550, value=300)
                
            # Add component button with more feedback
            if st.button("Add Component", type="primary", help="Add this component to your circuit"):
                if st.session_state.circuit_id:
                    with st.spinner(f"Adding {component_type}..."):
                        # Map friendly name to enum
                        type_map = {
                            "Resistor": "resistor",
                            "Capacitor": "capacitor",
                            "LED": "led",
                            "Voltage Source": "voltage_source",
                            "Ground": "ground"
                        }
                        
                        # Create component
                        component = {
                            "id": f"{type_map[component_type]}_{int(time.time())}",
                            "type": type_map[component_type],
                            "properties": properties,
                            "position": {"x": pos_x, "y": pos_y}
                        }
                        
                        # Add component via API
                        response = requests.post(
                            f"http://127.0.0.1:8000/circuits/{st.session_state.circuit_id}/components/", 
                            json=component
                        )
                        
                        if response.status_code == 200:
                            st.success(f"{component_type} added successfully!")
                            # Refresh circuit
                            response = requests.get(f"http://127.0.0.1:8000/circuits/{st.session_state.circuit_id}")
                            if response.status_code == 200:
                                st.session_state.circuit = response.json()
                        else:
                            st.error(f"Failed to add component: {response.text}")
                else:
                    st.error("Please create a new circuit first!")
        
        with tab3:
            # Simulation controls
            st.subheader("Simulation")
            sim_type = st.selectbox("Simulation Type", ["DC", "Transient", "AC"])
            
            sim_params = {}
            if sim_type == "Transient":
                sim_params["start_time"] = st.text_input("Start Time (s)", "0")
                sim_params["end_time"] = st.text_input("End Time (s)", "0.01")
                sim_params["step_size"] = st.text_input("Step Size (s)", "0.0001")
            elif sim_type == "AC":
                sim_params["start_freq"] = st.text_input("Start Frequency (Hz)", "1")
                sim_params["end_freq"] = st.text_input("End Frequency (Hz)", "1000000")
                sim_params["points_per_decade"] = st.text_input("Points per Decade", "10")
                
            if st.button("Run Simulation") and st.session_state.circuit_id:
                # Create simulation config
                config = SimulationConfig(
                    type=sim_type.lower(),
                    parameters=sim_params
                )
                
                # Run simulation via API
                response = requests.post(
                    f"http://127.0.0.1:8000/circuits/{st.session_state.circuit_id}/simulate/", 
                    json=config.dict()
                )
                
                if response.status_code == 200:
                    st.session_state.simulation_results = response.json()
                    st.success("Simulation complete!")
                else:
                    st.error(f"Simulation failed: {response.text}")
                
    # Main content area - split into two columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # AI Prompt area
        st.subheader("AI Circuit Designer")
        st.markdown("Describe the circuit you want to create in natural language.")
        
        prompt = st.text_area("Circuit Description", height=150, 
                               placeholder="Example: Create a simple LED circuit with a 330 ohm resistor and a 9V battery.")
        
        if st.button("Generate Circuit") and prompt:
            with st.spinner("Generating circuit..."):
                # Process with AI
                ai_request = AIRequest(
                    prompt=prompt,
                    circuit_id=st.session_state.circuit_id
                )
                
                response = requests.post(
                    "http://127.0.0.1:8000/ai/process/",
                    json=ai_request.dict()
                )
                
                if response.status_code == 200:
                    ai_response = response.json()
                    
                    # Display AI message
                    st.markdown("### AI Response")
                    st.write(ai_response["message"])
                    
                    # Display suggested actions
                    if ai_response.get("suggested_actions"):
                        st.markdown("### Suggested Actions")
                        for action in ai_response["suggested_actions"]:
                            st.markdown(f"- {action}")
                    
                    # Update circuit if one was generated
                    if ai_response.get("circuit"):
                        st.session_state.circuit = ai_response["circuit"]
                        st.session_state.circuit_id = ai_response["circuit"]["id"]
                        st.success("Circuit generated successfully!")
                else:
                    st.error(f"AI processing failed: {response.text}")
        
        # Show circuit image if available
        if st.session_state.circuit:
            st.subheader("Circuit Diagram")
            circuit_img = render_circuit_diagram(Circuit(**st.session_state.circuit))
            st.image(f"data:image/png;base64,{circuit_img}", use_column_width=True)
            
            # Show LED states (if any)
            led_components = [c for c_id, c in st.session_state.circuit["components"].items() 
                             if c["type"] == "led"]
            
            if led_components:
                st.subheader("LED Status")
                led_cols = st.columns(len(led_components))
                
                for i, led in enumerate(led_components):
                    with led_cols[i]:
                        color = led["properties"].get("color", "red")
                        on = led["state"].get("on", False) if "state" in led else False
                        led_img = render_led(color, on)
                        st.image(f"data:image/png;base64,{led_img}", width=100)
                        st.markdown(f"**{color.capitalize()} LED**: {'ON' if on else 'OFF'}")
    
    with col2:
        # Simulation results area
        st.subheader("Simulation Results")
        
        if st.session_state.simulation_results:
            results = st.session_state.simulation_results
            
            # Display plots if available
            if "plots" in results and results["plots"]:
                for plot_name, plot_data in results["plots"].items():
                    st.markdown(f"### {plot_name.replace('_', ' ').title()}")
                    st.image(f"data:image/png;base64,{plot_data}", use_column_width=True)
            
            # Display data table for detailed results
            if "data" in results:
                st.markdown("### Numerical Results")
                
                # Convert data to DataFrame for display
                data_dict = {}
                for key, values in results["data"].items():
                    if key != "time" and key != "frequency":
                        data_dict[key] = values
                
                if "time" in results["data"]:
                    data_dict["time"] = results["data"]["time"]
                elif "frequency" in results["data"]:
                    data_dict["frequency"] = results["data"]["frequency"]
                
                if data_dict:
                    df = pd.DataFrame(data_dict)
                    # Limit the number of rows displayed to prevent UI slowdown
                    st.dataframe(df.head(100))
                    
                    # Option to download full results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"simulation_results_{sim_type}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Run a simulation to see results here.")
            
        # Component list
        if st.session_state.circuit and st.session_state.circuit["components"]:
            st.subheader("Circuit Components")
            
            for comp_id, component in st.session_state.circuit["components"].items():
                with st.expander(f"{component['type'].capitalize()}: {comp_id}"):
                    st.json(component)
        else:
            st.info("Add components to the circuit to see them listed here.")

# Add this function to check if the server is running
def is_server_running():
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=1)
        return response.status_code == 200
    except:
        return False

# Entry point for the application
if __name__ == "__main__":
    main()
