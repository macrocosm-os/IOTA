module.exports = {
    apps: [{
        name: 'swarm-miner',
        script: './launch_miner.py',
        interpreter: '.venv/bin/python',
        cwd: './',
        watch: false,
        instances: 1,
        autorestart: true,
        max_restarts: 10,
        env: {
            NODE_ENV: 'production',
        },
        error_file: './logs/miner-error.log',
        out_file: './logs/miner-out.log',
        time: true,
    }]
};
