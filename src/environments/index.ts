import type { EnvironmentType } from "../types.ts";
import { BaseEnv } from "./base-env.ts";
import { LocalREPL, type LocalREPLOptions } from "./local-repl.ts";
import { DockerREPL, type DockerREPLOptions } from "./docker-repl.ts";

export * from "./base-env.ts";
export { LocalREPL, DockerREPL };
export type { LocalREPLOptions, DockerREPLOptions };

export type EnvKwargs = LocalREPLOptions | DockerREPLOptions;

export function getEnvironment(
  environment: EnvironmentType,
  kwargs: EnvKwargs,
): BaseEnv {
  switch (environment) {
    case "local":
      return new LocalREPL(kwargs as LocalREPLOptions);
    case "docker":
      return new DockerREPL(kwargs as DockerREPLOptions);
    default:
      throw new Error(
        `Unknown environment: ${environment as string}. Supported: ['local', 'docker']`,
      );
  }
}
