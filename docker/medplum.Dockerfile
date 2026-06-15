# Medplum Server (Open Source)
FROM medplum/medplum-server:4.5.2

# Set working directory to the standard monorepo root
WORKDIR /usr/src/medplum

# Copy config into the exact location expected by the server package
COPY docker/medplum.config.json packages/server/medplum.config.json

# Medplum server listens on 8103 by default
ENV PORT 8103
EXPOSE 8103

# Run the server
CMD ["node", "packages/server/dist/index.js"]
