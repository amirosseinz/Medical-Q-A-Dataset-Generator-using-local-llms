# ---- Build stage ----
FROM node:20-alpine AS build

# Ensure Node uses UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

WORKDIR /app

COPY package.json ./
RUN npm install --legacy-peer-deps

COPY . .
RUN npm run build

# ---- Production stage ----
FROM nginx:alpine

# Set locale for nginx process
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx config for SPA routing + API proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
