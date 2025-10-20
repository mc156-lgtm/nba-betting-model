# NBA Betting Model - Web App Deployment Guide

## 🌐 Live Demo

**Temporary Demo URL**: https://8502-ikgcsu6gn5d1yvjw9y3m5-26878163.manusvm.computer

⚠️ **Note**: This demo URL is temporary and will expire when the current session ends. Follow the deployment instructions below to host your own version.

## 📱 Web App Features

The Streamlit web interface provides:

### 🎯 Game Predictions
- Select any NBA matchup (home vs away team)
- Get predictions for:
  - **Spread**: Point spread with team advantages
  - **Total (Over/Under)**: Combined score prediction
  - **Win Probability**: Percentage chance each team wins
- **Edge Calculator**: Compare model predictions to sportsbook lines
- **Betting Recommendations**: AI-powered suggestions based on probabilities

### 👤 Player Props
- Enter player name and team
- Adjust expected minutes and games played
- Get predictions for:
  - Points (PTS)
  - Rebounds (REB)
  - Assists (AST)
  - Steals (STL)
  - Blocks (BLK)
- **Prop Betting Analysis**: Compare predictions to market lines

### 📊 Model Performance
- View accuracy metrics for all models
- See feature importance
- Understand model strengths and limitations

### ℹ️ About
- Learn how the models work
- Understand the technology stack
- Read disclaimers and best practices

## 🚀 Deployment Options

### Option 1: Run Locally (Easiest)

Perfect for personal use on your computer.

```bash
# Navigate to project directory
cd nba_betting_model

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the web app
streamlit run app.py

# Open browser to http://localhost:8501
```

**Pros**:
- Free
- Instant setup
- Full control
- No hosting costs

**Cons**:
- Only accessible on your computer
- Must keep terminal open

---

### Option 2: Streamlit Cloud (Free Hosting)

Deploy your app to the cloud for free!

**Steps**:

1. **Create GitHub Repository**
   ```bash
   cd nba_betting_model
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/nba-betting-model.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://YOUR_APP_NAME.streamlit.app`

**Pros**:
- Completely free
- Automatic HTTPS
- Easy updates (just push to GitHub)
- Shareable URL

**Cons**:
- Public by default (can make private with paid plan)
- Limited resources on free tier
- Sleeps after inactivity

**Cost**: Free forever

---

### Option 3: Heroku (Professional Hosting)

For more control and always-on hosting.

**Steps**:

1. **Install Heroku CLI**
   ```bash
   # Mac
   brew tap heroku/brew && brew install heroku
   
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   cd nba_betting_model
   heroku login
   heroku create nba-betting-model
   ```

3. **Add Procfile**
   ```bash
   echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
   ```

4. **Deploy**
   ```bash
   git init
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **Open your app**
   ```bash
   heroku open
   ```

**Pros**:
- Professional hosting
- Custom domains
- Always-on (with paid plan)
- Scalable

**Cons**:
- Costs $7/month for always-on
- More complex setup

**Cost**: $7/month (Eco Dynos)

---

### Option 4: DigitalOcean / AWS / Google Cloud

For full control and production deployment.

**DigitalOcean Droplet Setup**:

1. **Create Droplet** ($6/month)
   - Ubuntu 22.04
   - Basic plan ($6/month)
   - Choose datacenter region

2. **SSH into server**
   ```bash
   ssh root@YOUR_DROPLET_IP
   ```

3. **Install dependencies**
   ```bash
   apt update
   apt install -y python3.11 python3-pip nginx
   ```

4. **Upload your project**
   ```bash
   # On your local machine
   scp -r nba_betting_model root@YOUR_DROPLET_IP:/home/
   ```

5. **Install Python packages**
   ```bash
   cd /home/nba_betting_model
   pip3 install -r requirements.txt
   ```

6. **Run with systemd (auto-restart)**
   
   Create `/etc/systemd/system/nba-betting.service`:
   ```ini
   [Unit]
   Description=NBA Betting Model
   After=network.target
   
   [Service]
   User=root
   WorkingDirectory=/home/nba_betting_model
   ExecStart=/usr/bin/streamlit run app.py --server.port=8501
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable and start:
   ```bash
   systemctl enable nba-betting
   systemctl start nba-betting
   ```

7. **Setup Nginx reverse proxy**
   
   Create `/etc/nginx/sites-available/nba-betting`:
   ```nginx
   server {
       listen 80;
       server_name YOUR_DOMAIN_OR_IP;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```
   
   Enable site:
   ```bash
   ln -s /etc/nginx/sites-available/nba-betting /etc/nginx/sites-enabled/
   nginx -t
   systemctl restart nginx
   ```

8. **Access your app**: `http://YOUR_DROPLET_IP`

**Pros**:
- Full control
- Can run multiple services
- Professional setup
- Custom domain support

**Cons**:
- Requires server management
- More complex setup
- Need to handle security updates

**Cost**: $6/month (DigitalOcean Droplet)

---

### Option 5: PythonAnywhere

Simple Python hosting with web interface.

**Steps**:

1. Go to https://www.pythonanywhere.com
2. Create free account
3. Upload your files via web interface
4. Install dependencies in bash console
5. Configure web app settings
6. Your app will be at: `https://YOUR_USERNAME.pythonanywhere.com`

**Pros**:
- Easy web-based setup
- No command line needed
- Good for beginners

**Cons**:
- Limited CPU on free tier
- Streamlit may have issues (better for Flask)

**Cost**: Free tier available, $5/month for better performance

---

## 📊 Comparison Table

| Option | Cost | Setup Time | Best For | Always On | Custom Domain |
|--------|------|------------|----------|-----------|---------------|
| Local | Free | 2 min | Testing | No | No |
| Streamlit Cloud | Free | 10 min | Sharing | Auto-wake | No |
| Heroku | $7/mo | 30 min | Production | Yes | Yes |
| DigitalOcean | $6/mo | 1 hour | Full control | Yes | Yes |
| PythonAnywhere | $5/mo | 20 min | Simple hosting | Yes | Yes |

---

## 🔧 Configuration

### Environment Variables

For production, set these environment variables:

```bash
# Optional: Set custom port
export STREAMLIT_SERVER_PORT=8501

# Optional: Set custom title
export STREAMLIT_BROWSER_TITLE="NBA Betting Model"
```

### Custom Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

---

## 🔒 Security Considerations

### For Public Deployment

1. **Add Authentication** (Streamlit Cloud paid plan)
   - Restrict access to specific users
   - Use OAuth integration

2. **Rate Limiting**
   - Prevent abuse
   - Use Cloudflare or similar

3. **HTTPS**
   - Always use HTTPS in production
   - Free with Let's Encrypt

4. **Environment Secrets**
   - Don't commit API keys
   - Use Streamlit secrets management

### Example: Add Password Protection

```python
# Add to app.py
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "YOUR_PASSWORD":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    # Your app code here
    main()
```

---

## 📱 Mobile Optimization

Streamlit apps are mobile-responsive by default, but you can improve the experience:

```python
# Add to app.py
st.set_page_config(
    page_title="NBA Betting",
    page_icon="🏀",
    layout="wide",  # Use full width
    initial_sidebar_state="collapsed"  # Hide sidebar on mobile
)
```

---

## 🔄 Continuous Deployment

### Auto-deploy on Git push

**Streamlit Cloud**: Automatic

**Heroku**: Automatic with GitHub integration

**DigitalOcean**: Setup GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to DigitalOcean
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.DROPLET_IP }}
          username: root
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /home/nba_betting_model
            git pull
            systemctl restart nba-betting
```

---

## 📈 Monitoring & Analytics

### Add Google Analytics

```python
# Add to app.py
import streamlit.components.v1 as components

components.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR_GA_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR_GA_ID');
</script>
""", height=0)
```

---

## 🐛 Troubleshooting

### App won't start
- Check `requirements.txt` has all dependencies
- Verify Python version (3.11+)
- Check model files exist in `models/` directory

### Models not loading
- Ensure `.pkl` files are in correct location
- Check file permissions
- Verify joblib version matches training version

### Slow performance
- Use caching: `@st.cache_resource` for models
- Reduce feature count
- Use lighter models

### Port already in use
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run app.py --server.port 8502
```

---

## 🎨 Customization Ideas

### Add More Features

1. **Historical Performance Tracking**
   - Save predictions to database
   - Compare predictions vs actual results
   - Calculate ROI

2. **Betting Bankroll Management**
   - Kelly Criterion calculator
   - Unit size recommendations
   - Risk management tools

3. **Live Odds Integration**
   - Fetch real-time odds from APIs
   - Auto-calculate edges
   - Alert on high-value bets

4. **Multi-Game Parlays**
   - Combine multiple predictions
   - Calculate parlay odds
   - Risk/reward analysis

---

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Deployment Guide**: https://docs.streamlit.io/streamlit-community-cloud/get-started
- **Heroku Python**: https://devcenter.heroku.com/articles/getting-started-with-python
- **DigitalOcean Tutorials**: https://www.digitalocean.com/community/tutorials

---

## 💡 Recommended Setup

**For Personal Use**: Run locally or Streamlit Cloud (free)

**For Sharing with Friends**: Streamlit Cloud (free) or Heroku ($7/mo)

**For Professional/Commercial**: DigitalOcean ($6/mo) or AWS with custom domain

---

## ✅ Quick Start Checklist

- [ ] Download and extract `nba_betting_model.zip`
- [ ] Install Python 3.11+
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run locally: `streamlit run app.py`
- [ ] Test all features (Game Predictions, Player Props)
- [ ] Choose deployment option
- [ ] Deploy to chosen platform
- [ ] Share your URL!

---

**Need Help?** Check the README.md and documentation files included in the project.

**Want to Contribute?** The model is open for improvements - add real data, enhance features, or improve predictions!

