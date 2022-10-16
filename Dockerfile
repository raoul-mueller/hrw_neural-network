FROM dart:latest

WORKDIR /app
ENV PATH=$PATH:/root/.pub-cache/bin
EXPOSE 8080

RUN dart pub global activate webdev
RUN dart pub global activate stagehand

COPY pubspec.* ./
RUN dart pub get

COPY . .
RUN dart pub get --offline

CMD ["webdev", "serve", "--release", "--hostname", "0.0.0.0", "web:8080"]