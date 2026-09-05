import { describe, expect, it } from 'vitest';
import { composeSystemPrompt, normalizeInstruction } from './instructions';

describe('normalizeInstruction', () => {
  it('is null when the instruction says nothing', () => {
    expect(normalizeInstruction('   \n ')).toBeNull();
  });

  it('caps at the same length rapid-mac does', () => {
    expect(normalizeInstruction('x'.repeat(5000))?.length).toBe(4000);
  });
});

describe('composeSystemPrompt', () => {
  it('sends one layer as written, without precedence labels', () => {
    // The labels exist to RANK two layers. On one they are noise the model
    // has to read past.
    expect(composeSystemPrompt({ global: 'Be brief.', conversation: '' })).toBe('Be brief.');
    expect(composeSystemPrompt({ global: '', conversation: 'Use Rust.' })).toBe('Use Rust.');
  });

  it('ranks the conversation above the global default', () => {
    const prompt = composeSystemPrompt({ global: 'Be brief.', conversation: 'Be thorough.' });

    expect(prompt.indexOf('Be brief.')).toBeLessThan(prompt.indexOf('Be thorough.'));
    expect(prompt).toContain('HIGHEST USER PRIORITY');
  });

  it('is empty when neither layer says anything', () => {
    expect(composeSystemPrompt({ global: '  ', conversation: '' })).toBe('');
  });

  it('leads with the tool guidance', () => {
    // It describes the turn, not the user, so it must not be read as an
    // instruction the conversation layer outranks.
    const prompt = composeSystemPrompt({
      global: 'Be brief.',
      conversation: '',
      guidance: 'Answer from the results.',
    });

    expect(prompt.startsWith('Answer from the results.')).toBe(true);
  });
});
