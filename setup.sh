mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << 'EOF'
[server]
headless = true
enableCORS = true
enableXsrfProtection = true
EOF
