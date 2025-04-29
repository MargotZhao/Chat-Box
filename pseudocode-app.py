import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import numpy as np

st.set_page_config(
    page_title="LoRA Fine-Tuned Models Comparison",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #F0F2F6;
    }
    .chat-message.assistant {
        background-color: #E1EFFF;
    }
    .chat-message.original {
        background-color: #F7E4FF;
    }
    .chat-model-label {
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metrics-card {
        background-color: #FAFAFA;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .model-name {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🤖 LoRA Fine-Tuned Models Comparison")
st.markdown("""
This dashboard allows you to interact with three different fine-tuned language models.
You can compare their responses, evaluation metrics, and see the difference between pre and post fine-tuning.
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "response_times": {
            "model1": [],
            "model2": [],
            "model3": [],
            "model1_original": [],
            "model2_original": [],
            "model3_original": [],
        },
        "response_lengths": {
            "model1": [],
            "model2": [],
            "model3": [],
            "model1_original": [],
            "model2_original": [],
            "model3_original": [],
        }
    }

# Function to load models (in a real application, you would load your actual models)
@st.cache_resource
def load_models():
    # Simulating model loading
    # In a real application, you would load your models here
    
    # Example (commented out as it would depend on your specific models):
    # model1_original = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    # tokenizer1 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    # model1_finetuned = PeftModel.from_pretrained(model1_original, "./model1-lora/final")
    
    # Instead, we'll create placeholder functions that simulate model responses
    
    # This is just a placeholder - replace with actual model loading in your implementation
    return {
        "loaded": True,
        "message": "Models loaded successfully (placeholder)"
    }

# Load the models
models_loaded = load_models()

# Function to generate responses (simulated)
def generate_response(prompt, model_type):
    """
    Simulate generating a response from a model
    In a real application, you would use your actual models here
    """
    start_time = time.time()
    
    # Simulate different response behaviors for different models
    if model_type == "model1":
        time.sleep(0.2)  # Simulate processing time
        response = f"Model 1 (Fine-tuned) response: I'm the first fine-tuned model. {prompt}"
    elif model_type == "model2":
        time.sleep(0.3)  # Simulate processing time
        response = f"Model 2 (Fine-tuned) response: I'm the second fine-tuned model with more parameters. {prompt}"
    elif model_type == "model3":
        time.sleep(0.4)  # Simulate processing time
        response = f"Model 3 (Fine-tuned) response: I'm the third fine-tuned model with specialized training. {prompt}"
    elif model_type == "model1_original":
        time.sleep(0.15)  # Simulate processing time
        response = f"Model 1 (Original) response: Basic response without fine-tuning. {prompt}"
    elif model_type == "model2_original":
        time.sleep(0.25)  # Simulate processing time
        response = f"Model 2 (Original) response: Larger model without fine-tuning. {prompt}"
    elif model_type == "model3_original":
        time.sleep(0.35)  # Simulate processing time
        response = f"Model 3 (Original) response: Specialized model without fine-tuning. {prompt}"
        
    end_time = time.time()
    response_time = end_time - start_time
    
    # Record metrics
    st.session_state.metrics["response_times"][model_type].append(response_time)
    st.session_state.metrics["response_lengths"][model_type].append(len(response.split()))
    
    return response, response_time

# Create tabs for chat and metrics
tab1, tab2 = st.tabs(["Interactive Chat", "Performance Metrics"])

with tab1:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="chat-model-label">User</div>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            for model, response in message["content"].items():
                if "original" in model:
                    st.markdown(f"""
                    <div class="chat-message original">
                        <div class="chat-model-label">{model.replace('_', ' ').title()} (Before Fine-tuning)</div>
                        {response["text"]}
                        <div style="font-size: 0.8rem; margin-top: 0.5rem;">Response time: {response["time"]:.4f} seconds</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="chat-model-label">{model.replace('_', ' ').title()} (After Fine-tuning)</div>
                        {response["text"]}
                        <div style="font-size: 0.8rem; margin-top: 0.5rem;">Response time: {response["time"]:.4f} seconds</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_input_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100, max_chars=500)
        submitted = st.form_submit_button("Send")
        
        if submitted and user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get responses from all models
            responses = {}
            
            # Get responses from fine-tuned models
            for model_type in ["model1", "model2", "model3"]:
                response_text, response_time = generate_response(user_input, model_type)
                responses[model_type] = {"text": response_text, "time": response_time}
            
            # Get responses from original models
            for model_type in ["model1_original", "model2_original", "model3_original"]:
                response_text, response_time = generate_response(user_input, model_type)
                responses[model_type] = {"text": response_text, "time": response_time}
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": responses})
            
            # Force a rerun to display the new messages
            st.rerun()

with tab2:
    st.header("Performance Metrics")
    
    # Create two columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Response Time")
        st.write("Comparison of model response times")
        
        # Calculate average response times if we have data
        if any(len(times) > 0 for times in st.session_state.metrics["response_times"].values()):
            avg_times = {}
            for model, times in st.session_state.metrics["response_times"].items():
                if times:
                    avg_times[model] = sum(times) / len(times)
                else:
                    avg_times[model] = 0
            
            # Create DataFrame for plotting
            df_times = pd.DataFrame({
                'Model': list(avg_times.keys()),
                'Average Response Time (s)': list(avg_times.values())
            })
            
            # Create more readable model names for display
            df_times['Display Name'] = df_times['Model'].apply(
                lambda x: f"{x.replace('_original', '').title()} {'(Original)' if 'original' in x else '(Fine-tuned)'}"
            )
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(df_times['Display Name'], df_times['Average Response Time (s)'])
            
            # Color the bars
            for i, bar in enumerate(bars):
                if 'Original' in df_times['Display Name'].iloc[i]:
                    bar.set_color('#b19cd9')
                else:
                    bar.set_color('#6495ed')
            
            ax.set_ylabel('Average Response Time (seconds)')
            ax.set_title('Average Response Time by Model')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display the raw data
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            st.dataframe(df_times[['Display Name', 'Average Response Time (s)']])
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Send messages to generate response time metrics")
    
    with col2:
        st.subheader("Average Response Length")
        st.write("Comparison of model response lengths (word count)")
        
        # Calculate average response lengths if we have data
        if any(len(lengths) > 0 for lengths in st.session_state.metrics["response_lengths"].values()):
            avg_lengths = {}
            for model, lengths in st.session_state.metrics["response_lengths"].items():
                if lengths:
                    avg_lengths[model] = sum(lengths) / len(lengths)
                else:
                    avg_lengths[model] = 0
            
            # Create DataFrame for plotting
            df_lengths = pd.DataFrame({
                'Model': list(avg_lengths.keys()),
                'Average Word Count': list(avg_lengths.values())
            })
            
            # Create more readable model names for display
            df_lengths['Display Name'] = df_lengths['Model'].apply(
                lambda x: f"{x.replace('_original', '').title()} {'(Original)' if 'original' in x else '(Fine-tuned)'}"
            )
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(df_lengths['Display Name'], df_lengths['Average Word Count'])
            
            # Color the bars
            for i, bar in enumerate(bars):
                if 'Original' in df_lengths['Display Name'].iloc[i]:
                    bar.set_color('#b19cd9')
                else:
                    bar.set_color('#6495ed')
            
            ax.set_ylabel('Average Word Count')
            ax.set_title('Average Response Length by Model')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display the raw data
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            st.dataframe(df_lengths[['Display Name', 'Average Word Count']])
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Send messages to generate response length metrics")
    
    # Additional metrics section
    st.subheader("Model Specifications")
    
    # Create model cards with specifications
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("<div class='model-name'>Model 1</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Base Model**: GPT-2 (124M parameters)
        - **Fine-tuning Method**: LoRA
        - **LoRA Rank**: 8
        - **Training Data**: Custom dataset
        - **Memory Usage**: 512MB
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("<div class='model-name'>Model 2</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Base Model**: TinyLlama (1.1B parameters) 
        - **Fine-tuning Method**: LoRA
        - **LoRA Rank**: 16
        - **Training Data**: Custom dataset
        - **Memory Usage**: 1.2GB
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("<div class='model-name'>Model 3</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Base Model**: GPT-J (6B parameters)
        - **Fine-tuning Method**: LoRA
        - **LoRA Rank**: 32
        - **Training Data**: Custom dataset
        - **Memory Usage**: 3.5GB
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### How to use this dashboard:
    
    1. **Interactive Chat**: Type a message in the chat input and send it to see responses from all models.
    2. **Compare Models**: See how each model responds differently to the same input.
    3. **Before/After Comparison**: Compare each model's responses before and after fine-tuning.
    4. **Performance Metrics**: View response times, lengths, and other metrics to evaluate model performance.
    
    The dashboard automatically tracks performance metrics as you interact with the models.
    """)

# Instructions for implementation
st.sidebar.title("Implementation Notes")
st.sidebar.markdown("""
### To implement with your actual models:

1. Replace the `load_models()` function with code to load your actual fine-tuned models
2. Update the `generate_response()` function to use your models for inference
3. Adjust the model specifications in the metrics tab to match your models
4. Optional: Add additional metrics relevant to your specific models

This template is designed to be easily adaptable to your specific LoRA fine-tuned models.

### Required Libraries:
- streamlit
- pandas
- matplotlib
- transformers
- peft
- torch
""")