import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	contentText,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	type Model,
} from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { generateBranchSummary, prepareBranchEntries } from "../src/core/compaction/branch-summarization.ts";
import { compact, prepareCompaction } from "../src/core/compaction/compaction.ts";
import { boundSummaryFileOperations, formatFileOperations, stripFileOperations } from "../src/core/compaction/utils.ts";
import type { BranchSummaryEntry, CompactionEntry, SessionEntry } from "../src/core/session-manager.ts";

const maxChars = 2048 * 4;
const timestamp = new Date(0).toISOString();
const files = Array.from(
	{ length: 300 },
	(_, i) => `src/${String(i).padStart(4, "0")}/${"long-path/".repeat(8)}file.ts`,
);
const details = { readFiles: files, modifiedFiles: files.map((file) => `modified/${file}`) };
const model: Model<"anthropic-messages"> = {
	id: "test-model",
	name: "Test Model",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200000,
	maxTokens: 8192,
};

// The exact footer written before file lists were bounded.
const oldFooter = `\n\n<read-files>\n${details.readFiles.join("\n")}\n</read-files>\n\n<modified-files>\n${details.modifiedFiles.join("\n")}\n</modified-files>`;

function checkpoint(): CompactionEntry {
	return {
		type: "compaction",
		id: "checkpoint",
		parentId: null,
		timestamp,
		firstKeptEntryId: "checkpoint",
		tokensBefore: 100000,
		summary: `Keep the user's constraint.${oldFooter}`,
		details,
	};
}

function user(id: string, parentId: string): SessionEntry {
	return { type: "message", id, parentId, timestamp, message: { role: "user", content: id, timestamp: 0 } };
}

function summaryStream(prompts: string[]): StreamFn {
	return (_model, context) => {
		prompts.push(context.messages.map((message) => contentText(message.content, "")).join("\n"));
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() =>
			stream.push({ type: "done", reason: "stop", message: fauxAssistantMessage("New summary.") }),
		);
		return stream;
	};
}

describe("bounded summary file lists", () => {
	it("preserves empty and small lists exactly", () => {
		expect(formatFileOperations([], [])).toBe("");
		expect(formatFileOperations(["a.ts"], ["b.ts"])).toBe(
			"\n\n<read-files>\na.ts\n</read-files>\n\n<modified-files>\nb.ts\n</modified-files>",
		);
	});

	it("counts tags and whitespace at the exact budget boundary", () => {
		const overhead = "\n\n<read-files>\n\n</read-files>".length;
		const path = "a".repeat(maxChars - overhead);
		expect(formatFileOperations([path], [])).toHaveLength(maxChars);
		const tooLong = formatFileOperations([`${path}b`], []);
		expect(tooLong).toBe("\n\n<read-files>\n[... 1 more files omitted]\n</read-files>");
	});

	it.each([
		[details.readFiles, details.modifiedFiles],
		[details.readFiles, []],
		[[], details.modifiedFiles],
		[["README.md"], details.modifiedFiles],
		[details.readFiles, ["config.ts"]],
		[files.map((file) => `中文/📝/${file}`), details.modifiedFiles],
	])("keeps whole paths in order and reports all omissions within the shared budget", (readFiles, modifiedFiles) => {
		const before = structuredClone([readFiles, modifiedFiles]);
		const text = formatFileOperations(readFiles, modifiedFiles);
		expect(Math.ceil(text.length / 4)).toBeLessThanOrEqual(2048);
		for (const [tag, paths] of [
			["read-files", readFiles],
			["modified-files", modifiedFiles],
		] as const) {
			if (!paths.length) continue;
			const body = text.split(`<${tag}>\n`)[1].split(`\n</${tag}>`)[0];
			const lines = body.split("\n");
			const omitted = lines.at(-1)?.match(/^\[\.\.\. (\d+) more files omitted\]$/);
			if (omitted) lines.pop();
			expect(lines).toEqual(paths.slice(0, lines.length));
			expect(lines.length + Number(omitted?.[1] ?? 0)).toBe(paths.length);
			expect(lines.length).toBeGreaterThan(0);
		}
		expect([readFiles, modifiedFiles]).toEqual(before);
	});

	it("gives unused category space to the larger list", () => {
		const text = formatFileOperations(["README.md"], details.modifiedFiles);
		expect(text.length).toBeGreaterThan(maxChars * 0.9);
		expect(text).toContain("\nREADME.md\n");
	});

	it("removes only an exact terminal footer verified against details", () => {
		const entry = checkpoint();
		expect(stripFileOperations(entry)).toBe("Keep the user's constraint.");
		const bounded = boundSummaryFileOperations(entry);
		expect(bounded.length).toBeLessThanOrEqual("Keep the user's constraint.".length + maxChars);
		expect(stripFileOperations({ ...entry, summary: bounded })).toBe("Keep the user's constraint.");
		expect(boundSummaryFileOperations({ ...entry, summary: bounded })).toBe(bounded);

		for (const unchanged of [
			{ ...entry, fromHook: true },
			{ ...entry, details: undefined },
			{ ...entry, details: { readFiles: [null], modifiedFiles: [] } },
			{ ...entry, details: { readFiles: [], modifiedFiles: [] } },
			{ ...entry, summary: `${entry.summary}\nA narrative after the list.` },
		]) {
			expect(stripFileOperations(unchanged)).toBe(unchanged.summary);
			expect(boundSummaryFileOperations(unchanged)).toBe(unchanged.summary);
		}
		expect(entry.summary).toBe(`Keep the user's constraint.${oldFooter}`);
	});
});

describe("file lists across compaction boundaries", () => {
	it.each([false, true])("does not accumulate footers over repeated compactions (split turn: %s)", async (split) => {
		let previous = checkpoint();
		const prompts: string[] = [];
		for (let round = 0; round < 3; round++) {
			const first = user(`user-${round}`, previous.id);
			const operation: SessionEntry = {
				type: "message",
				id: `operation-${round}`,
				parentId: first.id,
				timestamp,
				message: {
					...fauxAssistantMessage(""),
					stopReason: "toolUse",
					content: [
						{ type: "toolCall", id: `edit-${round}`, name: "edit", arguments: { path: files[round] } },
						{ type: "toolCall", id: `write-${round}`, name: "write", arguments: { path: `new-${round}.ts` } },
					],
				},
			};
			const last: SessionEntry = split
				? {
						type: "message",
						id: `keep-${round}`,
						parentId: operation.id,
						timestamp,
						message: fauxAssistantMessage("Continue the same task."),
					}
				: user(`keep-${round}`, operation.id);
			const preparation = prepareCompaction([previous, first, operation, last], {
				enabled: true,
				reserveTokens: 16384,
				keepRecentTokens: 0,
			});
			expect(preparation?.isSplitTurn).toBe(split);
			expect(preparation?.previousSummary).not.toContain("<read-files>");
			expect(preparation?.previousSummary).not.toContain("<modified-files>");
			if (split) expect(preparation?.previousSummary).toContain("Keep the user's constraint.");
			if (!preparation) throw new Error("Expected compaction preparation");
			const result = await compact(
				preparation,
				model,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				summaryStream(prompts),
			);
			const expectedDetails = {
				readFiles: files.slice(round + 1),
				modifiedFiles: [
					...details.modifiedFiles,
					...files.slice(0, round + 1),
					...Array.from({ length: round + 1 }, (_, i) => `new-${i}.ts`),
				].sort(),
			};
			expect(result.details).toEqual(expectedDetails);
			expect(result.summary.match(/<read-files>/g)).toHaveLength(1);
			expect(result.summary.match(/<modified-files>/g)).toHaveLength(1);
			expect(result.summary.slice(result.summary.indexOf("\n\n<read-files>"))).toHaveLength(
				formatFileOperations(expectedDetails.readFiles, expectedDetails.modifiedFiles).length,
			);
			if (split) expect(result.summary).toContain("Keep the user's constraint.");
			previous = {
				...previous,
				...result,
				id: `checkpoint-${round}`,
				parentId: null,
				firstKeptEntryId: `checkpoint-${round}`,
			};
		}
		expect(prompts).toHaveLength(3);
		expect(prompts.every((prompt) => !prompt.includes("<read-files>") && !prompt.includes("<modified-files>"))).toBe(
			true,
		);
	});

	it("bounds a branch summary included in a compaction request, preserving narrative paths", async () => {
		const branch: BranchSummaryEntry = {
			type: "branch_summary",
			id: "branch",
			parentId: null,
			timestamp,
			fromId: "other",
			summary: `Investigate ${files.at(-1)}.${oldFooter}`,
			details,
		};
		const preparation = prepareCompaction([branch, user("keep", branch.id)], {
			enabled: true,
			reserveTokens: 16384,
			keepRecentTokens: 0,
		});
		if (!preparation) throw new Error("Expected compaction preparation");
		const prompts: string[] = [];
		await compact(preparation, model, undefined, undefined, undefined, undefined, undefined, summaryStream(prompts));
		expect(prompts[0]).toContain(`Investigate ${files.at(-1)}.`);
		expect(prompts[0]).toContain("more files omitted");
		expect(prompts[0]).not.toContain(oldFooter);
		expect(branch.summary).toContain(oldFooter);
	});

	it("bounds nested branch and compaction footers before branch budgeting and model input", async () => {
		const previous = checkpoint();
		const branchDetails = {
			readFiles: files.map((file) => `branch/${file}`),
			modifiedFiles: ["branch-change.ts"],
		};
		const branchFooter = `\n\n<read-files>\n${branchDetails.readFiles.join("\n")}\n</read-files>\n\n<modified-files>\nbranch-change.ts\n</modified-files>`;
		const branch: BranchSummaryEntry = {
			type: "branch_summary",
			id: "branch",
			parentId: previous.id,
			timestamp,
			fromId: "other",
			summary: `Branch narrative.${branchFooter}`,
			details: branchDetails,
		};
		const preparation = prepareBranchEntries([previous, branch], 5000);
		expect(preparation.messages).toHaveLength(2);
		expect(preparation.totalTokens).toBeLessThan(5000);
		const prompts: string[] = [];
		const result = await generateBranchSummary([previous, branch], {
			model,
			signal: new AbortController().signal,
			streamFn: summaryStream(prompts),
		});
		expect(prompts[0]).toContain("Keep the user's constraint.");
		expect(prompts[0]).toContain("Branch narrative.");
		expect(prompts[0]).not.toContain(oldFooter);
		expect(prompts[0]).not.toContain(branchFooter);
		expect(prompts[0]).toContain(`\n${files[0]}\n`);
		expect(prompts[0].length).toBeLessThan(2 * maxChars + 2000);
		// Branch tracking inherits branch details, while compaction-only file context stays in the prompt.
		expect(result.readFiles).toEqual(branchDetails.readFiles);
		expect(result.modifiedFiles).toEqual(branchDetails.modifiedFiles);
		expect(result.summary?.endsWith(formatFileOperations(branchDetails.readFiles, branchDetails.modifiedFiles))).toBe(
			true,
		);
		expect(result.summary?.match(/<read-files>/g)).toHaveLength(1);
	});
});
