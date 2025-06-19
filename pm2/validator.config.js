module.exports = {
    apps: [{
        name: 'swarm-validator',
        script: './launch_validator.py',
        interpreter: '.venv/bin/python',
        cwd: './',
        watch: false,
        instances: 1,
        autorestart: true,
        max_restarts: 10,
        env: {
            NODE_ENV: 'production',
        },
        error_file: './logs/validator-error.log',
        out_file: './logs/validator-out.log',
        time: true,
    }]
};
