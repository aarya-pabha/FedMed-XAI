# Stage 1: Build State (The "Everything" Stage)
FROM node:24 AS builder

WORKDIR /usr/src/medplum

# 1. Copy the ENTIRE cloned source
COPY medplum-build-env/ ./

# 2. Clean Install
RUN npm ci

# 3. Production Build
RUN npx turbo run build --filter=@medplum/server

# Stage 2: Runtime State (The "Lean" Stage)
FROM node:24-slim

ENV NODE_ENV=production
WORKDIR /usr/src/medplum

# 5. Copy production assets (dist and node_modules only)
COPY --from=builder /usr/src/medplum/package*.json ./
COPY --from=builder /usr/src/medplum/node_modules ./node_modules
COPY --from=builder /usr/src/medplum/packages/core/dist ./packages/core/dist
COPY --from=builder /usr/src/medplum/packages/server/dist ./packages/server/dist
COPY --from=builder /usr/src/medplum/packages/server/package.json ./packages/server/package.json
COPY --from=builder /usr/src/medplum/packages/definitions/dist ./packages/definitions/dist
COPY --from=builder /usr/src/medplum/packages/fhir-router/dist ./packages/fhir-router/dist
COPY --from=builder /usr/src/medplum/packages/fhirtypes/package.json ./packages/fhirtypes/package.json

# 6. Inject our validated config
COPY docker/medplum.config.json ./medplum.config.json

EXPOSE 8103

# 7. Final entrypoint using the official "env" merge strategy
CMD ["node", "--import", "/usr/src/medplum/packages/server/dist/otel/instrumentation.js", "/usr/src/medplum/packages/server/dist/index.js", "file:medplum.config.json,env"]
