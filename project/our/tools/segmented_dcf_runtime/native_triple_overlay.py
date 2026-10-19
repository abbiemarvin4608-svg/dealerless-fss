"""Private overlay for SCI's native secure bit-triple generator."""
from pathlib import Path


def native_triple_overlay(out: Path, sci: Path) -> None:
    relative = Path("Millionaire/bit-triple-generator.h")
    text = (sci / "src" / relative).read_text()
    enum_end = "};\n\nclass Triple {"
    if text.count(enum_end) != 1:
        raise ValueError("unsupported SCI triple-generator enum")
    text = text.replace(enum_end, '};\n#include "prepared_triples.h"\n\nclass Triple {', 1)
    signature = "  void generate(int party, uint8_t *ai, uint8_t *bi, uint8_t *ci,\n"
    if text.count(signature) != 1:
        raise ValueError("unsupported SCI raw triple generator")
    wrapper = '''  using sdcf_triple_method = TripleGenMethod;

  void generate(int party, uint8_t *ai, uint8_t *bi, uint8_t *ci,
                int num_triples, TripleGenMethod method, bool packed = false,
                int offset = 1) {
    if (!num_triples) return;
    if (sdcf_triples::intercept(otpack, ai, bi, ci, num_triples, int(method), packed, offset)) return;
    sdcf_preprocess::generation_begin(otpack);
    sdcf_preprocess::GenerationPurposeScope purpose;
    generate_native(party, ai, bi, ci, num_triples, method, packed, offset);
    sdcf_triples::completed();
  }

'''
    text = text.replace(signature, wrapper + "private:\n" + signature.replace("generate(", "generate_native("), 1)
    overload = "  void generate(int party, Triple *triples, TripleGenMethod method) {"
    if text.count(overload) != 1:
        raise ValueError("unsupported SCI Triple overload")
    text = text.replace(overload, "public:\n" + overload, 1)
    destination = out / "include_override" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
