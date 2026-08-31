package se.alipsa.jmlx.jinja.internal.parser;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateOptions;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.TemplateSyntaxException;
import se.alipsa.jmlx.jinja.internal.ast.Expression;
import se.alipsa.jmlx.jinja.internal.ast.Statement;
import se.alipsa.jmlx.jinja.internal.lexer.Token;
import se.alipsa.jmlx.jinja.internal.lexer.TokenType;

/** Recursive-descent parser ported one-to-one from upstream {@code parser.ts}. */
public final class Parser {
  private Parser() {}

  /**
   * Parses a token list into an AST.
   *
   * @param tokens the tokens produced by {@link se.alipsa.jmlx.jinja.internal.lexer.Lexer#tokenize}
   * @param options parse-time limits and syntax options
   * @return the parsed program
   */
  public static Statement.Program parse(List<Token> tokens, TemplateOptions options) {
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(options, "options");
    return new Cursor(tokens, options).parseProgram();
  }

  private static final class Cursor {
    private final List<Token> tokens;
    private final TemplateOptions options;
    private final SourceLocation endLocation;
    private int current;
    private int depth;

    Cursor(List<Token> tokens, TemplateOptions options) {
      this.tokens = List.copyOf(tokens);
      this.options = options;
      endLocation =
          tokens.isEmpty() ? new SourceLocation(0, 1, 1) : tokens.get(tokens.size() - 1).start();
    }

    Statement.Program parseProgram() {
      var body = new ArrayList<Statement>();
      var start = tokens.isEmpty() ? endLocation : tokens.get(0).start();
      while (current < tokens.size()) {
        body.add(parseAny());
      }
      return new Statement.Program(body, start);
    }

    Statement parseAny() {
      return switch (peek().type()) {
        case Comment -> {
          var t = next();
          yield new Statement.Comment(t.value(), t.start());
        }
        case Text -> parseText();
        // Statement bodies recursively re-enter parseAny, so statements count toward the same
        // depth budget as expression nesting and cannot exhaust the JVM stack.
        case OpenStatement -> nested(this::parseJinjaStatement);
        case OpenExpression -> parseJinjaExpression();
        default -> throw syntax("Unexpected token type: " + peek().type(), peek().start());
      };
    }

    Expression.StringLiteral parseText() {
      var t = expect(TokenType.Text, "Expected text token");
      return new Expression.StringLiteral(t.value(), t.start());
    }

    Statement parseJinjaStatement() {
      var start = expect(TokenType.OpenStatement, "Expected opening statement token").start();
      if (current >= tokens.size() || peek().type() != TokenType.Identifier) {
        throw syntax("Unknown statement, got " + typeHere(), locationHere());
      }
      var nameToken = peek();
      var name = nameToken.value();
      next();
      return switch (name) {
        case "set" -> parseSetStatement(start);
        case "if" -> {
          final var r = parseIfStatement(start);
          expect(TokenType.OpenStatement, "Expected {% token");
          expectIdentifier("endif");
          expect(TokenType.CloseStatement, "Expected %} token");
          yield r;
        }
        case "macro" -> {
          final var r = parseMacroStatement(start);
          expect(TokenType.OpenStatement, "Expected {% token");
          expectIdentifier("endmacro");
          expect(TokenType.CloseStatement, "Expected %} token");
          yield r;
        }
        case "for" -> {
          final var r = parseForStatement(start);
          expect(TokenType.OpenStatement, "Expected {% token");
          expectIdentifier("endfor");
          expect(TokenType.CloseStatement, "Expected %} token");
          yield r;
        }
        case "call" -> parseCallStatement(start);
        case "break" -> {
          expect(TokenType.CloseStatement, "Expected closing statement token");
          yield new Statement.Break(start);
        }
        case "continue" -> {
          expect(TokenType.CloseStatement, "Expected closing statement token");
          yield new Statement.Continue(start);
        }
        case "filter" -> parseFilterStatement(start);
        default -> throw syntax("Unknown statement type: " + name, nameToken.start());
      };
    }

    Expression parseJinjaExpression() {
      expect(TokenType.OpenExpression, "Expected opening expression token");
      var r = parseExpression();
      expect(TokenType.CloseExpression, "Expected closing expression token");
      return r;
    }

    Statement.SetStatement parseSetStatement(SourceLocation start) {
      var left = parseExpressionSequence(false);
      Expression value = null;
      var body = new ArrayList<Statement>();
      if (is(TokenType.Equals)) {
        next();
        value = parseExpressionSequence(false);
      } else {
        expect(TokenType.CloseStatement, "Expected %} token");
        while (!isStatement("endset")) {
          body.add(parseAny());
        }
        expect(TokenType.OpenStatement, "Expected {% token");
        expectIdentifier("endset");
      }
      expect(TokenType.CloseStatement, "Expected closing statement token");
      return new Statement.SetStatement(left, value, body, start);
    }

    Statement.If parseIfStatement(SourceLocation start) {
      final var test = parseExpression();
      expect(TokenType.CloseStatement, "Expected closing statement token");
      var body = new ArrayList<Statement>();
      var alternate = new ArrayList<Statement>();
      while (!isStatement("elif", "else", "endif")) {
        body.add(parseAny());
      }
      if (isStatement("elif")) {
        var elifStart = next().start();
        next();
        alternate.add(nested(() -> parseIfStatement(elifStart)));
      } else {
        if (isStatement("else")) {
          next();
          next();
          expect(TokenType.CloseStatement, "Expected closing statement token");
          while (!isStatement("endif")) {
            alternate.add(parseAny());
          }
        }
      }
      return new Statement.If(test, body, alternate, start);
    }

    Statement.Macro parseMacroStatement(SourceLocation start) {
      var name = parsePrimaryExpression();
      if (!(name instanceof Expression.Identifier id)) {
        throw syntax("Expected identifier following macro statement", name.location());
      }
      var args = parseArgs();
      expect(TokenType.CloseStatement, "Expected closing statement token");
      var body = new ArrayList<Statement>();
      while (!isStatement("endmacro")) {
        body.add(parseAny());
      }
      return new Statement.Macro(id, args, body, start);
    }

    Statement.For parseForStatement(SourceLocation start) {
      var variable = parseExpressionSequence(true);
      // Expression record names are pinned against upstream/vendor/src/ast.ts discriminators by
      // ParserTest.
      if (!(variable instanceof Expression.Identifier
          || variable instanceof Expression.TupleLiteral)) {
        throw syntax(
            "Expected identifier/tuple for the loop variable, got "
                + variable.getClass().getSimpleName()
                + " instead",
            variable.location());
      }
      if (!isIdentifier("in")) {
        throw syntax("Expected `in` keyword following loop variable", locationHere());
      }
      next();
      final var iterable = parseExpression();
      expect(TokenType.CloseStatement, "Expected closing statement token");
      var body = new ArrayList<Statement>();
      while (!isStatement("endfor", "else")) {
        body.add(parseAny());
      }
      var alternate = new ArrayList<Statement>();
      if (isStatement("else")) {
        next();
        next();
        expect(TokenType.CloseStatement, "Expected closing statement token");
        while (!isStatement("endfor")) {
          alternate.add(parseAny());
        }
      }
      return new Statement.For(variable, iterable, body, alternate, start);
    }

    Statement.CallStatement parseCallStatement(SourceLocation start) {
      final List<Expression> callerArgs = is(TokenType.OpenParen) ? parseArgs() : null;
      var callee = parsePrimaryExpression();
      if (!(callee instanceof Expression.Identifier)) {
        throw syntax("Expected identifier following call statement", callee.location());
      }
      final var callArgs = parseArgs();
      expect(TokenType.CloseStatement, "Expected closing statement token");
      var body = new ArrayList<Statement>();
      while (!isStatement("endcall")) {
        body.add(parseAny());
      }
      expect(TokenType.OpenStatement, "Expected '{%'");
      expectIdentifier("endcall");
      expect(TokenType.CloseStatement, "Expected closing statement token");
      return new Statement.CallStatement(
          new Expression.CallExpression(callee, callArgs, callee.location()),
          callerArgs,
          body,
          start);
    }

    Statement.FilterStatement parseFilterStatement(SourceLocation start) {
      Expression filter = parsePrimaryExpression();
      if (filter instanceof Expression.Identifier && is(TokenType.OpenParen)) {
        filter = parseCallExpression(filter);
      }
      expect(TokenType.CloseStatement, "Expected closing statement token");
      var body = new ArrayList<Statement>();
      while (!isStatement("endfilter")) {
        body.add(parseAny());
      }
      expect(TokenType.OpenStatement, "Expected '{%'");
      expectIdentifier("endfilter");
      expect(TokenType.CloseStatement, "Expected '%}'");
      return new Statement.FilterStatement(filter, body, start);
    }

    Expression parseExpressionSequence(boolean primary) {
      var values = new ArrayList<Expression>();
      values.add(primary ? parsePrimaryExpression() : parseExpression());
      boolean tuple = is(TokenType.Comma);
      while (is(TokenType.Comma)) {
        next();
        values.add(primary ? parsePrimaryExpression() : parseExpression());
      }
      return tuple ? new Expression.TupleLiteral(values, values.get(0).location()) : values.get(0);
    }

    Expression parseExpression() {
      return parseIfExpression();
    }

    Expression parseIfExpression() {
      var a = parseLogicalOrExpression();
      if (isIdentifier("if")) {
        next();
        var test = parseLogicalOrExpression();
        if (isIdentifier("else")) {
          next();
          return nested(() -> new Expression.Ternary(test, a, parseIfExpression(), a.location()));
        }
        return new Expression.SelectExpression(a, test, a.location());
      }
      return a;
    }

    Expression parseLogicalOrExpression() {
      var left = parseLogicalAndExpression();
      while (isIdentifier("or")) {
        var op = next();
        left =
            new Expression.BinaryExpression(op, left, parseLogicalAndExpression(), left.location());
      }
      return left;
    }

    Expression parseLogicalAndExpression() {
      var left = parseLogicalNegationExpression();
      while (isIdentifier("and")) {
        var op = next();
        left =
            new Expression.BinaryExpression(
                op, left, parseLogicalNegationExpression(), left.location());
      }
      return left;
    }

    Expression parseLogicalNegationExpression() {
      Expression.UnaryExpression right = null;
      while (isIdentifier("not")) {
        var op = next();
        right =
            nested(
                () ->
                    new Expression.UnaryExpression(
                        op, parseLogicalNegationExpression(), op.start()));
      }
      return right == null ? parseComparisonExpression() : right;
    }

    Expression parseComparisonExpression() {
      var left = parseAdditiveExpression();
      while (true) {
        Token op;
        if (isIdentifier("not", "in")) {
          op = new Token(TokenType.Identifier, "not in", peek().start());
          current += 2;
        } else {
          if (isIdentifier("in") || is(TokenType.ComparisonBinaryOperator)) {
            op = next();
          } else {
            break;
          }
        }
        left =
            new Expression.BinaryExpression(op, left, parseAdditiveExpression(), left.location());
      }
      return left;
    }

    Expression parseAdditiveExpression() {
      var left = parseMultiplicativeExpression();
      while (is(TokenType.AdditiveBinaryOperator)) {
        var op = next();
        left =
            new Expression.BinaryExpression(
                op, left, parseMultiplicativeExpression(), left.location());
      }
      return left;
    }

    Expression parseMultiplicativeExpression() {
      var left = parseTestExpression();
      while (is(TokenType.MultiplicativeBinaryOperator)) {
        var op = next();
        left = new Expression.BinaryExpression(op, left, parseTestExpression(), left.location());
      }
      return left;
    }

    Expression parseTestExpression() {
      var operand = parseFilterExpression();
      while (isIdentifier("is")) {
        next();
        boolean negate = isIdentifier("not");
        if (negate) {
          next();
        }
        var test = parsePrimaryExpression();
        if (!(test instanceof Expression.Identifier id)) {
          throw syntax("Expected identifier for the test", test.location());
        }
        operand = new Expression.TestExpression(operand, negate, id, operand.location());
      }
      return operand;
    }

    Expression parseFilterExpression() {
      var operand = parseCallMemberExpression();
      while (is(TokenType.Pipe)) {
        next();
        Expression filter = parsePrimaryExpression();
        if (!(filter instanceof Expression.Identifier)) {
          throw syntax("Expected identifier for the filter", filter.location());
        }
        if (is(TokenType.OpenParen)) {
          filter = parseCallExpression(filter);
        }
        operand = new Expression.FilterExpression(operand, filter, operand.location());
      }
      return operand;
    }

    Expression parseCallMemberExpression() {
      var member = parseMemberExpression(parsePrimaryExpression());
      return is(TokenType.OpenParen) ? parseCallExpression(member) : member;
    }

    Expression parseCallExpression(Expression callee) {
      return nested(
          () -> {
            Expression result =
                new Expression.CallExpression(callee, parseArgs(), callee.location());
            result = parseMemberExpression(result);
            return is(TokenType.OpenParen) ? parseCallExpression(result) : result;
          });
    }

    List<Expression> parseArgs() {
      expect(TokenType.OpenParen, "Expected opening parenthesis for arguments list");
      var result = parseArgumentsList();
      expect(TokenType.CloseParen, "Expected closing parenthesis for arguments list");
      return result;
    }

    List<Expression> parseArgumentsList() {
      return nested(
          () -> {
            var args = new ArrayList<Expression>();
            while (!is(TokenType.CloseParen)) {
              Expression arg;
              if (is(TokenType.MultiplicativeBinaryOperator) && peek().value().equals("*")) {
                var start = next().start();
                arg = new Expression.SpreadExpression(parseExpression(), start);
              } else {
                arg = parseExpression();
                if (is(TokenType.Equals)) {
                  next();
                  if (!(arg instanceof Expression.Identifier id)) {
                    throw syntax("Expected identifier for keyword argument", arg.location());
                  }
                  arg =
                      new Expression.KeywordArgumentExpression(
                          id, parseExpression(), id.location());
                }
              }
              args.add(arg);
              if (is(TokenType.Comma)) {
                next();
              }
            }
            return args;
          });
    }

    Expression parseMemberExpressionArgumentsList(SourceLocation openBracket) {
      return nested(
          () -> {
            var slices = new ArrayList<Expression>();
            boolean slice = false;
            while (!is(TokenType.CloseSquareBracket)) {
              if (is(TokenType.Colon)) {
                slices.add(null);
                next();
                slice = true;
              } else {
                slices.add(parseExpression());
                if (is(TokenType.Colon)) {
                  next();
                  slice = true;
                }
              }
            }
            if (slices.isEmpty()) {
              throw syntax(
                  "Expected at least one argument for member/slice expression", locationHere());
            }
            if (slice) {
              if (slices.size() > 3) {
                throw syntax("Expected 0-3 arguments for slice expression", locationHere());
              }
              var start = slices.get(0);
              var stop = slices.size() > 1 ? slices.get(1) : null;
              var step = slices.size() > 2 ? slices.get(2) : null;
              return new Expression.SliceExpression(start, stop, step, openBracket);
            }
            return slices.get(0);
          });
    }

    Expression parseMemberExpression(Expression object) {
      while (is(TokenType.Dot) || is(TokenType.OpenSquareBracket)) {
        var op = next();
        boolean computed = op.type() == TokenType.OpenSquareBracket;
        Expression property;
        if (computed) {
          property = parseMemberExpressionArgumentsList(op.start());
          expect(TokenType.CloseSquareBracket, "Expected closing square bracket");
        } else {
          property = parsePrimaryExpression();
          if (!(property instanceof Expression.Identifier
              || property instanceof Expression.IntegerLiteral)) {
            throw syntax(
                "Expected identifier or integer following dot operator", property.location());
          }
        }
        object = new Expression.MemberExpression(object, property, computed, object.location());
      }
      return object;
    }

    Expression parsePrimaryExpression() {
      var token = next();
      return switch (token.type()) {
        case NumericLiteral -> number(token);
        case StringLiteral -> {
          var value = new StringBuilder(token.value());
          while (is(TokenType.StringLiteral)) {
            value.append(next().value());
          }
          yield new Expression.StringLiteral(value.toString(), token.start());
        }
        case Identifier -> new Expression.Identifier(token.value(), token.start());
        case OpenParen ->
            nested(
                () -> {
                  var r = parseExpressionSequence(false);
                  // Upstream accidentally uses a double-quoted JavaScript string here, leaving
                  // this template placeholder literal in its diagnostic.
                  expect(
                      TokenType.CloseParen,
                      "Expected closing parenthesis, got ${tokens[current].type} instead.");
                  return r;
                });
        case OpenSquareBracket ->
            nested(
                () -> {
                  var values = new ArrayList<Expression>();
                  while (!is(TokenType.CloseSquareBracket)) {
                    values.add(parseExpression());
                    if (is(TokenType.Comma)) {
                      next();
                    }
                  }
                  next();
                  return new Expression.ArrayLiteral(values, token.start());
                });
        case OpenCurlyBracket ->
            nested(
                () -> {
                  var values = new ArrayList<Expression.ObjectEntry>();
                  while (!is(TokenType.CloseCurlyBracket)) {
                    var key = parseExpression();
                    expect(
                        TokenType.Colon, "Expected colon between key and value in object literal");
                    values.add(new Expression.ObjectEntry(key, parseExpression()));
                    if (is(TokenType.Comma)) {
                      next();
                    }
                  }
                  next();
                  return new Expression.ObjectLiteral(values, token.start());
                });
        default -> throw syntax("Unexpected token: " + token.type(), token.start());
      };
    }

    Expression number(Token token) {
      return token.value().contains(".")
          ? new Expression.FloatLiteral(Double.parseDouble(token.value()), token.start())
          : new Expression.IntegerLiteral(Double.parseDouble(token.value()), token.start());
    }

    Token peek() {
      if (current >= tokens.size()) {
        throw syntax("Unexpected end of template", endLocation);
      }
      return tokens.get(current);
    }

    Token next() {
      var t = peek();
      current++;
      return t;
    }

    Token expect(TokenType type, String error) {
      if (current >= tokens.size()) {
        throw syntax("Parser Error: " + error + ". End of template !== " + type + ".", endLocation);
      }
      var t = tokens.get(current++);
      if (t.type() != type) {
        throw syntax("Parser Error: " + error + ". " + t.type() + " !== " + type + ".", t.start());
      }
      return t;
    }

    boolean is(TokenType... types) {
      if (current + types.length > tokens.size()) {
        return false;
      }
      for (var i = 0; i < types.length; i++) {
        if (tokens.get(current + i).type() != types[i]) {
          return false;
        }
      }
      return true;
    }

    boolean isIdentifier(String... names) {
      if (current + names.length > tokens.size()) {
        return false;
      }
      for (var i = 0; i < names.length; i++) {
        if (tokens.get(current + i).type() != TokenType.Identifier
            || !names[i].equals(tokens.get(current + i).value())) {
          return false;
        }
      }
      return true;
    }

    boolean isStatement(String... names) {
      return current + 1 < tokens.size()
          && tokens.get(current).type() == TokenType.OpenStatement
          && tokens.get(current + 1).type() == TokenType.Identifier
          && java.util.Arrays.asList(names).contains(tokens.get(current + 1).value());
    }

    void expectIdentifier(String name) {
      if (!isIdentifier(name)) {
        throw syntax("Expected " + name, locationHere());
      }
      current++;
    }

    SourceLocation locationHere() {
      return current < tokens.size() ? tokens.get(current).start() : endLocation;
    }

    String typeHere() {
      return current < tokens.size() ? tokens.get(current).type().toString() : "end of template";
    }

    <T> T nested(Supplier<T> production) {
      if (++depth > options.maxAstDepth()) {
        --depth;
        throw new TemplateRenderException(
            "AST depth exceeds the configured limit of " + options.maxAstDepth(),
            ErrorCategory.RESOURCE_LIMIT,
            locationHere());
      }
      try {
        return production.get();
      } finally {
        depth--;
      }
    }

    TemplateSyntaxException syntax(String message, SourceLocation location) {
      return new TemplateSyntaxException(message, location);
    }
  }
}
