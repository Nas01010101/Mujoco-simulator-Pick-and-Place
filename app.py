"""
Robot Pick-and-Place: Imitation Learning Demo
A clean Streamlit interface for training and comparing IL algorithms.
"""
import streamlit as st
import plotly.graph_objects as go
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Robot Pick-and-Place",
    page_icon=None,
    layout="wide"
)

# Modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%); }
    
    h1 { color: #1a1a2e; font-weight: 700; }
    h2, h3 { color: #16213e; font-weight: 600; }
    
    .hero-title { 
        font-size: 2.5rem; 
        font-weight: 700; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle { color: #666; font-size: 1.1rem; margin-bottom: 2rem; }
    
    .metric-row { display: flex; gap: 1.5rem; margin: 2rem 0; flex-wrap: wrap; }
    .metric { 
        background: white; 
        padding: 1.5rem 2rem; 
        border-radius: 12px; 
        text-align: center; 
        flex: 1; 
        min-width: 140px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.5rem; }
    
    .algo-card { 
        background: white; 
        padding: 1.5rem; 
        border-radius: 12px; 
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid;
        transition: transform 0.2s;
    }
    .algo-card:hover { transform: translateX(4px); }
    .algo-card.bc { border-color: #667eea; }
    .algo-card.dagger { border-color: #10b981; }
    .algo-card.gail { border-color: #8b5cf6; }
    .algo-card.diffusion { border-color: #f59e0b; }
    
    .algo-card h4 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #1a1a2e; }
    .algo-card p { margin: 0; color: #666; font-size: 0.85rem; line-height: 1.5; }
    
    .section-header { 
        font-size: 1.25rem; 
        font-weight: 600; 
        color: #1a1a2e; 
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stSidebar"] { background: #fafbfc; }
    
    .limitations-section {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
    }
    .limitations-section h4 { color: #2d3748; margin-bottom: 1rem; }
    .limitations-section p { color: #4a5568; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem; }
    .limitations-section li { color: #4a5568; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


def load_data():
    try:
        with open('assets/data/training_stats.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'epochs': list(range(1, 201)),
            'loss': [0.3 * (0.95 ** i) + 0.05 for i in range(200)],
            'final_loss': 0.11, 'num_demos': 100, 'num_transitions': 15000,
            'learning_rate': 0.0005, 'batch_size': 64
        }


# Sidebar
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", ["Overview", "Train", "Compare", "Results", "About"], label_visibility="collapsed")

data = load_data()

# ============== OVERVIEW ==============
if page == "Overview":
    st.markdown('<p class="hero-title">Robot Pick-and-Place</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Teaching a UR5e robot arm to manipulate objects through imitation learning</p>', unsafe_allow_html=True)
    
    # Metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric"><div class="metric-value">{data['num_demos']}</div><div class="metric-label">Expert Demos</div></div>
        <div class="metric"><div class="metric-value">{data['final_loss']:.2f}</div><div class="metric-label">Final Loss</div></div>
        <div class="metric"><div class="metric-value">4</div><div class="metric-label">Algorithms</div></div>
        <div class="metric"><div class="metric-value">7D</div><div class="metric-label">Observation Space</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo Videos (Emojis removed)
    st.markdown('<p class="section-header">Demo Videos</p>', unsafe_allow_html=True)
    videos = sorted([f for f in os.listdir('assets/videos') if f.endswith('.mp4')])
    if videos:
        cols = st.columns(min(3, len(videos)))
        for i, video in enumerate(videos[:3]):
            with cols[i]:
                st.video(f'assets/videos/{video}')
                st.caption(video.replace('.mp4', '').replace('_', ' ').title())
    else:
        st.info("No demo videos found. Run `python sim/expert_demo.py` to generate.")
    
    # Algorithms (Emojis removed)
    st.markdown('<p class="section-header">Algorithms</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        <div class="algo-card bc">
            <h4>Behavior Cloning</h4>
            <p>Supervised learning that directly maps observations to actions. Fast training, simple implementation.</p>
        </div>
        <div class="algo-card dagger">
            <h4>DAgger</h4>
            <p>Interactive learning with online expert corrections. Reduces distribution shift by training on visited states.</p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="algo-card gail">
            <h4>GAIL</h4>
            <p>Adversarial imitation using a discriminator. Learns implicit reward function from demonstrations.</p>
        </div>
        <div class="algo-card diffusion">
            <h4>Diffusion Policy</h4>
            <p>Generative modeling via denoising diffusion. Captures multi-modal action distributions.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Project Limitations Section (Lucid & Mature)
    st.markdown('''
    <div class="limitations-section">
        <h4>Project Limitations & Context</h4>
        <p>While this project demonstrates fundamental concepts of imitation learning, it presents a simplified environment compared to real-world deployment challenges.</p>
        <ul>
            <li><strong>Simulation Gap:</strong> The physics simulation (MuJoCo) provides perfect state information and zero-latency control. Real-world robotics must contend with noisy sensors, mechanical imperfections, and significant control delays.</li>
            <li><strong>Data Scarcity:</strong> The models here attempt to learn from a small dataset (~50-100 episodes). Robust industrial systems typically require thousands of diverse demonstrations or sim-to-real transfer with domain randomization.</li>
            <li><strong>Generalization:</strong> The learned policies are specialized for this specific bin location and object type. They lack the generalization capabilities of foundation models (like RT-1 or Octo) which can handle novel objects and instructions.</li>
            <li><strong>Distribution Shift:</strong> Simple Behavior Cloning policies suffer from compounding errors when deviating from the expert's trajectory. While DAgger and Diffusion Policies mitigate this, long-horizon tasks remain a significant research challenge.</li>
        </ul>
        <p><em>This codebase serves as an educational sandbox for understanding algorithm mechanics, rather than a production-ready control system.</em></p>
    </div>
    ''', unsafe_allow_html=True)

# ============== TRAIN ==============
elif page == "Train":
    st.title("Train Policy")
    st.markdown("Select an algorithm and configure hyperparameters.")
    
    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox("Algorithm", ["Behavior Cloning", "DAgger", "GAIL", "Diffusion Policy"])
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.0005)
        batch_size = st.select_slider("Batch Size", options=[32, 64, 128, 256], value=64)
    with col2:
        num_epochs = st.slider("Epochs", min_value=10, max_value=200, value=50, step=10)
        hidden_size = st.select_slider("Hidden Size", options=[64, 128, 256, 512], value=256)
    
    algo_descriptions = {
        "Behavior Cloning": "**Supervised Learning** — Trains a neural network to directly map observations to actions by minimizing MSE loss against expert demonstrations.",
        "DAgger": "**Interactive Learning** — Dataset Aggregation iteratively improves the policy by rolling out the current policy and querying the expert for corrections.",
        "GAIL": "**Adversarial Learning** — Uses a discriminator to distinguish expert from policy behavior. The policy learns to fool the discriminator.",
        "Diffusion Policy": "**Generative Modeling** — Uses denoising diffusion to model the action distribution. Captures multi-modal behaviors."
    }
    st.info(algo_descriptions[algorithm])
    
    if st.button("Start Training", type="primary", use_container_width=True):
        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        try:
            if algorithm == "Behavior Cloning":
                from ml.train_live import train_bc_live
                trainer = train_bc_live(learning_rate=learning_rate, batch_size=batch_size, 
                                       num_epochs=num_epochs, hidden_size=hidden_size)
            elif algorithm == "DAgger":
                from ml.dagger import train_dagger_live
                trainer = train_dagger_live(learning_rate=learning_rate, batch_size=batch_size,
                                           num_iterations=max(1, num_epochs // 20), hidden_size=hidden_size)
            elif algorithm == "GAIL":
                from ml.gail import train_gail_live
                trainer = train_gail_live(learning_rate=learning_rate, batch_size=batch_size,
                                         num_epochs=num_epochs, hidden_size=hidden_size)
            else:
                from ml.diffusion_policy import train_diffusion_live
                trainer = train_diffusion_live(learning_rate=learning_rate, batch_size=batch_size,
                                              num_epochs=num_epochs, hidden_size=hidden_size)
            
            for update in trainer:
                progress_bar.progress(update['progress'] / 100)
                loss_val = update.get('loss', update.get('gen_loss', 0))
                status_text.text(f"Epoch {update.get('epoch', update.get('iteration', 0))} | Loss: {loss_val:.4f}")
                
                losses = update.get('losses', update.get('losses_g', []))
                if losses:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(1, len(losses)+1)), y=losses,
                                            mode='lines', line=dict(color='#667eea', width=2)))
                    fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', height=300,
                                     margin=dict(l=40, r=20, t=20, b=40))
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key="live_training_chart")
                
                if update.get('done'):
                    st.success("Training complete! Model saved.")
        except Exception as e:
            st.error(f"Error: {e}")

# ============== COMPARE ==============
elif page == "Compare":
    st.title("Algorithm Comparison")
    
    st.markdown("""
    <div class="info-banner">
        <p>Each algorithm has different trade-offs. Behavior Cloning is fastest but may struggle with distribution shift. 
        DAgger addresses this through interactive learning. GAIL learns implicit rewards. Diffusion Policy handles multi-modal actions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    comparison = {
        'Algorithm': ['Behavior Cloning', 'DAgger', 'GAIL', 'Diffusion Policy'],
        'Type': ['Supervised', 'Interactive', 'Adversarial', 'Generative'],
        'Training Time': ['~30s', '~2min', '~1min', '~45s'],
        'Complexity': ['Low', 'Medium', 'High', 'High']
    }
    st.table(comparison)
    
    st.info("Run `python ml/eval.py` to measure actual success rates. Results vary based on training.")

# ============== RESULTS ==============
elif page == "Results":
    st.title("Results")
    
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric"><div class="metric-value">50</div><div class="metric-label">Test Episodes</div></div>
        <div class="metric"><div class="metric-value">{data['num_transitions']:,}</div><div class="metric-label">Training Samples</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Episode Videos</p>', unsafe_allow_html=True)
    videos = sorted([f for f in os.listdir('assets/videos') if f.endswith('.mp4')])
    if videos:
        selected = st.selectbox("Select episode", videos)
        st.video(f'assets/videos/{selected}')

# ============== ABOUT ==============
else:
    st.title("About This Project")
    
    st.markdown("""
    <div class="info-banner">
        <p>This project explores imitation learning for robot manipulation. A UR5e robot learns to pick up 
        a red cube and place it in a green bin by imitating expert demonstrations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        <div class="algo-card bc">
            <h4>The Task</h4>
            <p>Pick-and-place: Grasp a cube from a random position on a table and drop it into a target bin.</p>
        </div>
        <div class="algo-card dagger">
            <h4>Technologies</h4>
            <p>MuJoCo physics simulation, PyTorch neural networks, Jacobian-based IK control</p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="algo-card gail">
            <h4>Observation Space</h4>
            <p>7D: Gripper position (3), Cube position (3), Gripper state (1)</p>
        </div>
        <div class="algo-card diffusion">
            <h4>Action Space</h4>
            <p>4D: Target position (3), Grip command (1)</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("### References")
    st.markdown("### Recommended Reading")
    st.markdown("""
    **1. Foundations & Theory**  
    [**Underactuated Robotics: Imitation Learning**](https://underactuated.mit.edu/output_feedback.html) — *Russ Tedrake (MIT)*  
    A rigorous yet accessible textbook chapter explaining the control theory behind Behavior Cloning and its stability challenges.

    **2. DAgger & Distribution Shift**  
    [**CS285: Deep Reinforcement Learning**](https://rail.eecs.berkeley.edu/deeprlcourse/) — *Sergey Levine (UC Berkeley)*  
    Lecture notes that intuitively explain why "error compounds" in BC and how algorithm like DAgger (Dataset Aggregation) fix it.

    **3. Modern Generative Approaches**  
    [**Diffusion Policy Project Page**](https://diffusion-policy.cs.columbia.edu/) — *Chi et al. (Columbia/Toyota)*  
    Highly visual explanation of using diffusion models for robot control. Excellent figures and video examples of the state-of-the-art.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("MuJoCo + PyTorch + Streamlit")
